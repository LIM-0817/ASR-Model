import torch
import torch.nn as nn 

from models import TransformerListener, Attention, Speller


# 정의한 Listener, Attender, Speller을 조립하여 ASR model 정의
class ASRModel(torch.nn.Module):
    def __init__(self, batch_size, input_size, embed_dim, lstm_step): # embed_dim = listener_hidden_size*2
        super().__init__()

        # listener, attender, speller 정의
        self.listener = TransformerListener(input_size = input_size)
        self.attender = Attention(embed_dim, embed_dim, embed_dim)
        self.speller = Speller(batch_size, self.attender, embed_dim, lstm_step)

    def forward(self, x,lx,y=None,teacher_forcing_ratio=1):
        # 1. Listener을 이용해 speech data Encoding 
        encoder_outputs, encoder_len = self.listener(x,lx)

        # 2. Decoding step을 위해서 key,value 값 계산
            ## encoder의 key, value는 변하지 않는 값이므로 미리 이렇게 계산하는 것.
        self.attender.set_key_value(encoder_outputs)   

        # 3. attention을 이용한 decoding step
        raw_outputs, attention_plots = self.speller(encoder_output = encoder_outputs, encoder_len = encoder_len, y=y,teacher_forcing_ratio=teacher_forcing_ratio)

        return raw_outputs, attention_plots