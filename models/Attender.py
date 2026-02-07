import torch
import torch.nn as nn 
import math


# Listener과 Speller 사이의 attention context 계산을 위한 Attention class
class Attention(torch.nn.Module):
    def __init__(self, listener_hidden_size,
                speller_hidden_size,
                projection_size):
        super().__init__()
        # k, v, q 생성을 위한 linear 층 생성
        self.KW = nn.Linear(listener_hidden_size, projection_size)
        self.VW = nn.Linear(listener_hidden_size, projection_size)
        self.QW = nn.Linear(speller_hidden_size, projection_size)

    # set_key_value 함수 실행 시 key, value값 계산
    def set_key_value(self, encoder_outputs):   
        self.key = self.KW(encoder_outputs)
        self.value = self.VW(encoder_outputs) 

    # attention context 계산 함수
        ## 해당 attention context는 speller에서 input과 concatenate되어 LSTM으로 들어감.
    def compute_context(self, decoder_context):
        query = self.QW(decoder_context) # (batch, dim)
        query = query.unsqueeze(1) # (batch, 1, dim)

        # 1. attention weight 계산 = k*q batch matmul & softmax
        dim = query.size(-1)
        raw = torch.bmm(query, self.key.transpose(-1, -2)) / math.sqrt(dim) # (batch, 1, seq)
        raw = raw.squeeze(1) # (batch, seq)
        attention_weights = torch.nn.functional.softmax(raw, dim = -1)

        # 2. attention context = attention weight과 v의 batch matmul
        attention_context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
        attention_context.squeeze(1)
        
        if torch.isnan(attention_weights).any():
            print("NaN in attention_weights")

        # 분포 체크
        print("attn max:", attention_weights.max().item(),
            "min:", attention_weights.min().item(),
            "mean:", attention_weights.mean().item())

        return attention_context, attention_weights