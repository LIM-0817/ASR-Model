import torch
import torch.nn as nn 
import numpy as np

from .Attender import Attention
from config import DEVICE
from config import VOCAB_MAP


# 5. Speller class
class Speller(torch.nn.Module):
    def __init__(self, batch_size, attender:Attention, embed_dim, lstm_step_count): # lstm_step = 한번에 lstm 몇개 사용할건지
        super(). __init__()
        self.attend = attender # Attention object in speller
        self.max_timesteps = 550 # Max timesteps
        self.embed_dim = embed_dim
        self.lstm_step_count = lstm_step_count
        self.batch_size = batch_size

        # 각 token embedding하기 위해 생성
        self.embedding = nn.Embedding(31, embed_dim) 

        # LSTM cell의 sequence 생성
        self.lstm_cells = nn.ModuleList() 
        ## LSTM layer의 처음 층은 고정
        self.lstm_cells.append(nn.LSTMCell(embed_dim*2, embed_dim)) 
        ## lstm_step_count의 수만큼 lstm cell 생성
        for i in range(self.lstm_step_count-1):
            self.lstm_cells.append(nn.LSTMCell(embed_dim, embed_dim)) #nn.ModuleList는 list이므로 append이용

        # CDN 층 생성
        ## dim을 맞춰주기 위한(embed_dim*2 -> embed_dim)
        self.output_to_char = nn.Linear(embed_dim*2, embed_dim)# Linear module to convert outputs to correct hidden size (Optional: TO make dimensions match)
        self.activation = nn.ReLU()# Check which activation is suggested
        self.char_prob = nn.Linear(embed_dim, 31) #각 단어에 대한 확률 logit으로 만들기 위해 전체 단어장 크기인 31로 dim을 바꿈.

        # Weight tying: '단어 -> 임베딩 벡터'와 '임베딩 벡터 -> 단어 logit'은 정확히 반대방향의 변환
        # 따라서 weight tying, 즉 두 과정의 가중치를 동일하게 만들어줘서 파라미터수는 줄이고 학습을 효율화
        self.char_prob.weight = self.embedding.weight


    # lstm_step 함수는 한 번 lstm 층을 통과시킨 다음, 다음 LSTM에 넣을
    def lstm_step(self, input_word, hidden_state): 
        # hidden_state = 이전 step의 [(h_t_cell1, c_t_cell1), (h_t_cell2,c_t_cell2)]의 tuple
        input_embed = input_word # input_word = embedding을 거쳐서 (batch_size, embed_dim) 형태

        # 1. 각 layer의 hidden_states를 저장하기 위한 list 생성
            ## nn.LSTMCell을 이용하면 각 단계에서 (h_t_cell1, c_t_cell1), (h_t_cell2, c_t_cell2)의 tuple이 필요함.
        next_hidden_states = []
        
        # 2. lstm에 input_embed 넣어주기
        for i, cell in enumerate(self.lstm_cells):
            # layer 1에서는 embedding한 token과 이전 time step의 layer 1 hidden_state을 받아서 넣음.
            if i == 0:  
                hidden_tuple = self.lstm_cells[i](input_embed, hidden_state[0]) 
                next_hidden_states.append(hidden_tuple)
            
            # layer 2는 방금 전의 layer 1 h_t와 이전 time step의 layer 2의 hidden_state을 받아서 넣어줌
            else:       
                hidden_tuple = self.lstm_cells[i](next_hidden_states[i-1][0], hidden_state[i])
                next_hidden_states.append(hidden_tuple)
        
        # 해당 time_step의 모든 layer에 대한 hidden_state 튜플이 모두 들어있음.
        lstm_input = next_hidden_states 

        return lstm_input

    def CDN(self, lstm_out):
        # LSTM 결과물을 init에서 만든 CDN층에 통과시키기
        out = self.output_to_char(lstm_out)
        out = self.activation(out)
        out = self.char_prob(out)
        
        return out

    def forward (self, encoder_output,  y=None, teacher_forcing_ratio=1): # y = groundtruth를 의미
        raw_outputs = []        # 결과로 나온 logit (vocab_size,)
        attention_plot = []     # attention weight plotting을 위해 모아 놓을 리스트

        # 1. attention context 초기화
        attn_context = torch.zeros((self.batch_size, self.embed_dim), device = DEVICE)
        
        # 2. 맨처음 문자는 <sos>로 모두 채움.
        output_symbol = torch.full((self.batch_size, 1), VOCAB_MAP['<sos>'], device=DEVICE)

        # 3. 각 seq의 timestep을 설정하기
        if y is None: # groundtruth가 없다면?
            timesteps = self.max_timesteps
            teacher_forcing_ratio = 0 # tf ratio는 당연히 0
        else:
            timesteps = y.size(1) 

        # 4. lstm_step 개수 만큼 hidden state list에 hidden state을 초기화
        hidden_states_list = [] # hidden_states_list 초기화: [(h1, c1), (h2, c2), ...]
        for i in range(self.lstm_step_count):
            h1 = torch.zeros((self.batch_size, self.embed_dim), device=DEVICE)
            c1 = torch.zeros((self.batch_size, self.embed_dim), device=DEVICE)
            hidden_states_list.append((h1, c1))

        
        # 5. 이제 진짜 speller의 feed forward loop시작
        for t in range(timesteps):
            # 1) teacher forcing을 할지 말지 결정
            p = np.random.rand(1) 

            ## 0일 때는 <sos>가 들어가므로 다음 seq 추측 불가. 따라서 teacher forcing
            ## t = timesteps - 1, 즉 for 문의 마지막 t의 경우에는 teacherforcing 사용 x
            if p < teacher_forcing_ratio and 0 < t < timesteps - 1: 
                input_symbol = y[:, t]  # 정답을 feed
            else:
                input_symbol = output_symbol[:, -1].clone().detach()   # 이전 단계의 예측을 feed

            # 2) input값을 embedding
            char_embed = self.embedding(input_symbol) 

            # 3) attention context와 embedded input을 concat
            lstm_input = torch.cat((char_embed, attn_context), dim = -1)

            # 4) lstm_step에 통과 시키고, 꺼낸 hidden_state로 hidden_states_list를 갱신
            new_hidden = self.lstm_step(lstm_input, hidden_states_list) 
            hidden_states_list = new_hidden

            # 5) 가장 마지막 layer의 hidden state를 이용해서 attention context를 계산
                ## attender의 key and value는 ASR model에서 계산될 것
            attn_context, attn_weights = self.attend.compute_context(hidden_states_list[self.lstm_step_count-1][0]) 

            # 6) CDN 층에 통과
                ## cdn에 들어갈 input = 이전 layer의 hidden state 및 이를 이용해서 새로 계산한 attention context를 concat
            cdn_input = torch.cat((hidden_states_list[self.lstm_step_count-1][0], attn_context), dim = -1) 
            raw_pred = self.CDN(cdn_input) 

            # 7) greedy 이용해서 예측 output 찾아내기
            output = torch.argmax(raw_pred, dim = -1) # raw_pred에서 나온 것 중 확률 가장 높은 애를 채택 (batch, Vocab_size=31)
            output_symbol = torch.cat((output_symbol, output.unsqueeze(dim = 1)), dim = -1)

            raw_outputs.append(raw_pred) # Loss 값 계산을 위해 사용
            attention_plot.append(attn_weights) # attention plot을 위해서 list에 append

        # 6. 각 time step에서 나온 결과들을 합치기
        attention_plot = torch.stack(attention_plot, dim=1)
        raw_outputs = torch.stack(raw_outputs, dim=1) #stack을 통해 (batch, seq, embed_dim)

        return raw_outputs, attention_plot