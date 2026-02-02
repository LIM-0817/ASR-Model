import torch
import torch.nn as nn 
import math 

from config import DEVICE


# 1. PositionalEncoding class
class PositionalEncoding(torch.nn.Module):
    def __init__(self, projection_size, max_seq_len= 5000):
        super().__init__()
        # Attention is all you need 논문의 positional encoding(sin, cos이용)
        ## 위치 인코딩 정보를 담을 행렬 초기화
        pe = torch.zeros(max_seq_len, projection_size)
        
        ## position = [0, 1, ... max_seq_len - 1]
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        ## div_term: 10000^(2i/projection_size)
        div_term = torch.exp(torch.arange(0, projection_size, 2).float() * (-math.log(10000.0) / projection_size))

        ## 짝수 번째에 sin, 홀수 번째에는 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        ## pe가 학습하지 않도록 buffer로 저장
        self.register_buffer('pe', pe)


    def forward(self, x):   # x = (batch, seq, dim)
        x = x.to(DEVICE)
        return x + self.pe[:x.size(1), :]



# 2. TransformerEncoder class
class TransformerEncoder(torch.nn.Module):
    def __init__(self, projection_size, num_heads = 1, dropout= 0.0):
        super().__init__()

        ## key, value, query의 가중치 생성
        self.KW = nn.Linear(projection_size, projection_size) 
        self.VW = nn.Linear(projection_size, projection_size) 
        self.QW = nn.Linear(projection_size, projection_size) 

        ## MultiheadAttention 사용
        self.attention = nn.MultiheadAttention(projection_size, num_heads = num_heads, batch_first=True)
        
        ## LayerNorm 정의
        self.bn1 = nn.LayerNorm(projection_size)
        self.bn2 = nn.LayerNorm(projection_size)

        ## Feed Forward Neural Network
        self.MLP = nn.Sequential(  
                        nn.Linear(projection_size, projection_size*4),
                        nn.ReLU(),
                        nn.Linear(4*projection_size, projection_size),
                        nn.ReLU()
                    ) # 주의. nn.Sequential 내에는 nn. , 즉 class만 이용가능


    def forward(self, x, src_key_padding_mask = None):  # x = (batch, seq, dim)
        # 1. k, v, q 계산
        x = x.to(DEVICE)
        key = self.KW(x) 
        value = self.VW(x) 
        query = self.QW(x) 

        # 2. Attention 계산
        out1, _ = self.attention(query, key, value, key_padding_mask=src_key_padding_mask)
        ## residual connection 
        out1 = x + out1
        ## Batch Normalization
        out1 = self.bn1(out1)

        # 3. Feed Forward Network에 통과시키기
        out2 = self.MLP(out1)
        ## residual connection 
        out2 = out1 + out2
        ## Batch Normalization
        out2 = self.bn2(out2)

        return out2