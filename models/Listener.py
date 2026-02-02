import torch
import torch.nn as nn 

from .layers import PositionalEncoding
from .layers import TransformerEncoder
from config import DEVICE


# TransformerListner class
    # mfcc를 LSTM(초기 특징 추출)에 통과시켜 나온 데이터를 CNN에 넣어 MFCC의 지역적인 특징을 추출한 다음, 
    # 이후 이걸 self-attention 시키는 구조임
class TransformerListener(torch.nn.Module):
    def __init__(self,
                input_size,
                base_lstm_layers        = 1,
                pblstm_layers           = 1,
                listener_hidden_size    = 64,
                n_heads                 = 8,
                tf_blocks               = 1):
        super().__init__()

        # LSTM layer
        self.base_lstm = nn.LSTM(
                input_size, 
                listener_hidden_size, 
                base_lstm_layers, 
                batch_first = True, 
                bidirectional = True
            )   # 결과 dim = listener_hidden_Size*2
        
        # 1D CNN layer 생성(시간 순의 MFCC의 지역적 특징을 뽑아내기 위해 1d CNN이용)
            ## kernel size를 3로, padding = 1
            ## inputsize - kernelsize + 2*padding을 통해 output_size도 input_size로 맞춤
        self.embedding = nn.Conv1d(listener_hidden_size*2, listener_hidden_size*2, kernel_size = 3, stride = 1, padding = 1) 

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
        _, unsorted_indices = torch.sort(sorted_indices)
        output = output[unsorted_indices]
        output_lengths = output_lengths[unsorted_indices]
        
        # 6. 1d CNN 통과
            ## Conv1d에 넣을 때 (batch, dim, seq) 순서로. 따라서 transpose 후 다시 원래대로 돌림.
        output = self.embedding(output.transpose(-1, -2))   #, Conv1d에 넣을 때 (batch, dim, seq 순서로 넣음)
        output = output.transpose(-1, -2) # (batch, seq, dim)
        
        # 7. Transformer용 패딩 마스크 생성
            ## 실제 값이 아닌 패딩 부분은 True인 mask를 생성
            ## output_lengths는 1차원 텐서(batch,)로, 각 배치마다의 문장 길이를 담고 있음. 
            ## torch.arange(max_len).unsqueeze(0)는 (1, seq), output_lengths.unsqueeze(1)은 (batch, 1)
            ## 각각이 (batch, seq)으로 맞춰져서 broadcasting으로 비교
        max_len = output.shape[1]
        mask = (torch.arange(max_len, device=x.device).unsqueeze(0) >= output_lengths.unsqueeze(1)) 
        
        # 8. Transformer에 넣기 전 positional encoding 
        output  = self.positional_encoding(output) 

        # 9. Transformer에 통과
            ## mask를 넣어서 패딩은 연산에서 제외하도록 함. 
        output = self.transformer_encoder(output, src_key_padding_mask = mask)

        return output, output_lengths