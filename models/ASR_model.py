import torch
import torch.nn as nn 

from models import TransformerListener, Attention, Speller
from config import config


# 정의한 Listener, Attender, Speller을 조립하여 ASR model 정의
class ASRModel(torch.nn.Module):
    def __init__(self,
                listener_hidden_size, 
                batch_size, 
                input_size, 
                embed_dim, 
                lstm_step,
                decode_mode='greedy'
                ): # embed_dim = listener_hidden_size*2
        super().__init__()

        self.decode_mode = decode_mode

        # listener, attender, speller 정의
        self.listener = TransformerListener(input_size = input_size)
        self.attender = Attention(listener_hidden_size*2, embed_dim, embed_dim)
        self.speller = Speller(self.attender, embed_dim, lstm_step)

    def forward(self, x, lx, y=None, teacher_forcing_ratio=1):
        # 1. Listener을 이용해 speech data Encoding 
        encoder_outputs, encoder_len = self.listener(x,lx)

        if self.decode_mode == 'beam':
            
            ### beam search 시 슈퍼배치로 인해 encoder_outputs와 encoder_len을
            ### 그대로 attention 계산에 사용하면 차원이 안맞으므로, 
            ### b*k 상태로 만들어 준 뒤에 set_key_value 함수 실행
            topk = config["top_k"]
            encoder_len = encoder_len.repeat_interleave(topk, dim=0)
            encoder_outputs = encoder_outputs.repeat_interleave(topk, dim=0)


        # 2. Decoding step을 위해서 key,value 값 계산 
        self.attender.set_key_value(encoder_outputs)   

        # 3. decoding step
        ## greedy: train 시 사용
        if self.decode_mode == 'greedy':
            raw_predictions, attention_plots = self.speller(
                encoder_output = encoder_outputs, 
                encoder_len = encoder_len, 
                y=y,
                teacher_forcing_ratio=teacher_forcing_ratio
            )
            return raw_predictions, attention_plots

        ## beam search: test(validation도 가능)에서 주로 사용
        elif self.decode_mode == 'beam':
            predictions = self.speller.beam_search(
                encoder_output = encoder_outputs, 
                encoder_len = encoder_len
            )
            return predictions, None
        
                
        ## config의 decode mode가 잘못된 경우 raise error
        else:
            raise ValueError("'greedy' 혹은 'beam'을 입력하시오")