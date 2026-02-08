import torch
import torch.nn as nn 

from .layers import PositionalEncoding
from .layers import TransformerEncoder
from config import DEVICE, config


# TransformerListner class
    # mfcc를 LSTM(초기 특징 추출)에 통과시켜 나온 데이터를 CNN에 넣어 MFCC의 지역적인 특징을 추출한 다음, 
    # 이후 이걸 self-attention 시키는 구조임
class TransformerListener(torch.nn.Module):
    def __init__(self,
                input_size,
                base_lstm_layers = 1,
                listener_hidden_size = config["listener_hidden_size"],
                n_heads = 8,
            ):
        super().__init__()

        # LSTM layer 1 - for pblstm
        self.base_lstm = nn.LSTM(
                input_size, 
                listener_hidden_size, 
                base_lstm_layers, 
                batch_first = True, 
                bidirectional = True    # 결과 dim = listener_hidden_Size*2
            )   
        
        # LSTM layer 2 - for pblstm 
        self.middle_lstm = nn.LSTM(
                listener_hidden_size*4, # base_lstm의 결과 
                listener_hidden_size,
                1, 
                batch_first = True,
                bidirectional = True,
            )

        # LSTM layer 3 - for pblstm
        self.top_lstm = nn.LSTM(
                listener_hidden_size*4,
                listener_hidden_size,
                1,
                batch_first = True,
                bidirectional = True,
            )       
        
        # 1D CNN layer 생성(시간 순의 MFCC의 지역적 특징을 뽑아내기 위해 1d CNN이용)
            ## kernel size를 3로, padding = 1
            ## inputsize - kernelsize + 2*padding을 통해 output_size도 input_size로 맞춤
        self.embedding = nn.Conv1d(listener_hidden_size*2, listener_hidden_size*2, kernel_size = 3, stride = 2, padding = 1) 

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(listener_hidden_size*2)

        # Transformer Encoder
        self.transformer_encoder = TransformerEncoder(listener_hidden_size*2, n_heads)

    
    def forward(self, x, x_len):    # x = (batch, seq, dim)
        x = x.to(DEVICE)

        # 1. 효율적인 연산을 위해 seq의 길이를 내림차순으로 정리
        x_len_sorted, sorted_indices = torch.sort(x_len, descending=True)
        x_sorted = x[sorted_indices]

        # !! x_len_sorted를 gpu가 아닌 cpu로 이동 후 int64로 타입 변환 !!
        x_length_cpu = x_len_sorted.cpu().long() 

        # 2. padding이 되어있던 x_sorted를 packing(LSTM을 통과 시킬 때 패딩 값들까지 불필요하게 연산 되지 않도록)
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x_sorted, x_length_cpu, batch_first=True)

        # 3. LSTM통과
        lstm_out, _ = self.base_lstm(x_packed)

        # 4. Unpacking -> 다시 패딩 복구
        output, output_lengths  = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True) # (Batch, seq, embed_dim)
        output_lengths = output_lengths.to(x.device) # 결과 길이는 다시 gpu에 올리기
        
        # 5. 언패킹 이후 원래 배치 순서대로 텐서를 복구(나중에 Label과의 짝 맞춰야하므로 꼭 필요함.)
        _, unsorted_indices = torch.sort(sorted_indices)    # sorted indices를 정렬하여 해당 순서대로 배열하면 
        output = output[unsorted_indices]                   # 다시 원래대로 돌아옴.(0, 1, 2... 순으로 배열되므로)
        output_lengths = output_lengths[unsorted_indices]

        # 6. pblstm (first stage - base_lstm output to middle_lstm input)
        ## 1) output의 seq가 짝수인 경우 pad를 1개 줄이기
        if output.size(1) % 2 == 1:
            output = output[:, :-1, :]

        ## 2) output의 seq길이 1/2배, dim 2배
        b, s, d = output.size()
        input_to_middle = output.contiguous().view([b, s//2, d*2])
        output_lengths = output_lengths // 2

        ## 3) middle_lstm에 넣어줌.
        input_to_middle_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input_to_middle,
                output_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
                )

        middle_lstm_out_packed, _ = self.middle_lstm(input_to_middle_packed)
        middle_lstm_out, middle_lstm_out_lengths = torch.nn.utils.rnn.pad_packed_sequence(middle_lstm_out_packed, batch_first=True)
        output_lengths = middle_lstm_out_lengths.to(x.device)

        # 7. pblstm (2nd stage - middle_lstm output to top_lstm input)
        ## 1) output의 seq가 짝수인 경우 pad를 1개 줄이기
        if middle_lstm_out.size(1) % 2 == 1:
            middle_lstm_out = middle_lstm_out[:, :-1, :]

        ## 2) output의 seq길이 1/2배, dim 2배
        b, s, d = middle_lstm_out.size()
        input_to_top = middle_lstm_out.contiguous().view([b, s//2, d*2])
        output_lengths = output_lengths // 2 # halve again

        ## 3) top_lstm에 다시 넣어줌.
        input_to_top_packed = torch.nn.utils.rnn.pack_padded_sequence(
                input_to_top,
                output_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
                )

        lstm2_out_packed, _ = self.top_lstm(input_to_top_packed)
        lstm2_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm2_out_packed, batch_first=True)
        
        # 8. 1d CNN 통과
            ## Conv1d에 넣을 때 (batch, dim, seq) 순서로. 따라서 transpose 후 다시 원래대로 돌림.
        output_emb = self.embedding(lstm2_out.transpose(-1, -2))   #, Conv1d에 넣을 때 (batch, dim, seq 순서로 넣음)
        output_emb = output_emb.transpose(-1, -2) # (batch, seq, dim)

        ## CNN에 stride와 pad를 넣었으니 output_lengths 보정 필요
        ## 실제 공식: L_out = (L_in + 2*pad - kernel)/stride + 1
        output_lengths = torch.clamp(output_lengths//2, max=output_emb.size(1))
        
        # 9. encoder용 패딩 마스크 생성
            ## 실제 값이 아닌 패딩 부분은 True인 mask를 생성
            ## output_lengths는 1차원 텐서(batch,)로, 각 배치마다의 문장 길이를 담고 있음. 
            ## torch.arange(max_len).unsqueeze(0)는 (1, seq), output_lengths.unsqueeze(1)은 (batch, 1)
            ## 각각이 (batch, seq)으로 맞춰져서 broadcasting으로 비교
        max_len = output_emb.shape[1]
        mask = (torch.arange(max_len, device=x.device).unsqueeze(0) >= output_lengths.unsqueeze(1)) 
        
        # 10. encoder에 넣기 전 positional encoding 
        output_pos  = self.positional_encoding(output_emb) 

        # 11. encoder에 통과
            ## mask를 넣어서 패딩은 연산에서 제외하도록 함. 
        output_fin = self.transformer_encoder(output_pos, src_key_padding_mask = mask)

        return output_fin, output_lengths