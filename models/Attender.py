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
        self.key = self.KW(encoder_outputs)     # (batch, max_len, )
        self.value = self.VW(encoder_outputs) 

    # attention context 계산 함수
        ## 해당 attention context는 speller에서 input과 concatenate되어 LSTM으로 들어감.
    def compute_context(self, decoder_context, encoder_len):
        query = self.QW(decoder_context) # (batch, dim)
        query = query.unsqueeze(1) # (batch, 1, dim)

        # 1. attention weight 계산 = k*q batch matmul & softmax
        dim = query.size(-1)
        raw = torch.bmm(query, self.key.transpose(-1, -2)) / math.sqrt(dim) # (batch, 1, seq)
        raw = raw.squeeze(1) # (batch, seq)

        # 2. masking 로직 추가(없어서 패딩 구간에 attention weight을 주고 있었을 듯)
        max_len = self.key.size(1)  # key는 패딩도 포함하고 있으므로, 실제 seq길이가 아니라 max_len이 길이임

        ## torch.arange(max_len) -> (max_len, )
        ## torch.arange(max_len).expand(len(encoder_len), -1) -> (batch, max_len)
            ### expand는 같은 행렬을 복붙해서 열이나 행을 늘려주는 역할을 한다.
            ### expand로 [0, 1, 2, ... , max_len - 1]의 행렬을 len(encoder_len) = batch만큼 늘려 (batch, max_len)행렬을 만듦
        ## encoder_len = (batch, ) 이므로 unsqueeze(1)로 (batch, 1)로 바꿔서
        ## broadcasting을 통해 mask 생성
        mask = torch.arange(max_len, device=self.key.device).expand(len(encoder_len), max_len) >= encoder_len.unsqueeze(1)
        
        ## autocast를 사용하고 있는데 ,-1e9는 float16으로는 표현 불가
        ## torch.finfo(raw.dtype).min을 쓰면 type에 맞는 가장 작은 수가 할당됨
        small_num = torch.finfo(raw.dtype).min
        raw = raw.masked_fill_(mask, small_num)  # 작은 수 써서 softmax 시 0이 됨.

        attention_weights = torch.nn.functional.softmax(raw, dim = -1)

        # 2. attention context = attention weight과 v의 batch matmul
        attention_context = torch.bmm(attention_weights.unsqueeze(1), self.value).squeeze(1)
        
        return attention_context, attention_weights