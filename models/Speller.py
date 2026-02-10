import torch
import torch.nn as nn 
import numpy as np

from .Attender import Attention
from config import DEVICE, config, VOCAB_MAP


# 5. Speller class
class Speller(torch.nn.Module):
    def __init__(self, attender:Attention, embed_dim, lstm_step_count): # lstm_step = 한번에 lstm 몇개 사용할건지
        super(). __init__()
        self.attend = attender # Attention object in speller
        self.max_timesteps = 550 # Max timesteps
        self.embed_dim = embed_dim
        self.lstm_step_count = lstm_step_count

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
        input_embed = input_word # input_word = embedding을 거쳐서 (batch, embed_dim) 형태

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

    # 1. greedy search 실행
    def forward(self, encoder_output, encoder_len, y=None, teacher_forcing_ratio=1): # y = groundtruth를 의미
        raw_outputs = []        # 결과로 나온 logit (vocab_size,)
        attention_plot = []     # attention weight plotting을 위해 모아 놓을 리스트

        batch = encoder_output.size(0)

        # 1. attention context 초기화
        attn_context = torch.zeros((batch, self.embed_dim), device = DEVICE)
        
        # 2. 맨처음 문자는 <sos>로 모두 채움.
        output_symbol = torch.full((batch, 1), VOCAB_MAP['<sos>'], device=DEVICE)

        # 3. 각 seq의 timestep을 설정하기
        if y is None: # groundtruth가 없다면?
            timesteps = self.max_timesteps
            teacher_forcing_ratio = 0 # tf ratio는 당연히 0
        else:
            timesteps = y.size(1) 

        # 4. lstm_step 개수 만큼 hidden state list에 hidden state을 초기화
        hidden_states_list = [] # hidden_states_list 초기화: [(h1, c1), (h2, c2), ...]
        for i in range(self.lstm_step_count):
            h1 = torch.zeros((batch, self.embed_dim), device=DEVICE)
            c1 = torch.zeros((batch, self.embed_dim), device=DEVICE)
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
            attn_context, attn_weights = self.attend.compute_context(hidden_states_list[self.lstm_step_count-1][0], encoder_len) 

            # 6) CDN 층에 통과
                ## cdn에 들어갈 input = 이전 layer의 hidden state 및 이를 이용해서 새로 계산한 attention context를 concat
            cdn_input = torch.cat((hidden_states_list[self.lstm_step_count-1][0], attn_context), dim = -1) 
            raw_pred = self.CDN(cdn_input) # (batch, Vocab_size=31)

            # 7) greedy 이용해서 예측 output 찾아내기
            output = torch.argmax(raw_pred, dim = -1) # raw_pred에서 나온 것 중 확률 가장 높은 애를 채택
            output_symbol = torch.cat((output_symbol, output.unsqueeze(dim = 1)), dim = -1)

            raw_outputs.append(raw_pred) # Loss 값 계산을 위해 사용
            attention_plot.append(attn_weights) # attention plot을 위해서 list에 append

        # 6. 각 time step에서 나온 결과들을 합치기
        attention_plot = torch.stack(attention_plot, dim=1)
        raw_outputs = torch.stack(raw_outputs, dim=1) #stack을 통해 (batch, seq, embed_dim)

        return raw_outputs, attention_plot
    

    # 2. beam search 실행
    def beam_search(self, encoder_output, encoder_len): 
        raw_outputs = []        
        attention_plot = []     
        batch = encoder_output.size(0)
        topk = config["top_k"]
        vocab_size = len(VOCAB_MAP) 

        # 1. 배치와 빔을 합쳐서 슈퍼 배치를 만듦. 즉 0차원의 크기를 batch*topk
            ## repeat() &repeat_interleave()
            ## - [A, B] 라는 텐서를 3번 복사해 늘리고 싶다면, 
            ## - repeat() -> [A, B, A, B, A, B] = 통째로 반복
            ## - repeat() -> [A, A, A, B, B, B] = 제자리 반복
            ## 같은 배치 내의 데이터는 연속적으로 topk 개수 만큼 있어야 다루기 편하므로, repeat_interleave를 이용함.
        
        ## test.py에 보면 encoder_output과 encoder_len은 이미 슈퍼배치가 되서 들어오고 있음.
        encoder_output = encoder_output  # (b*k, seq, dim)
        encoder_len = encoder_len        # (b*k, )

        # 2. lstm_step 개수 만큼 hidden state list에 hidden state을 초기화
        hidden_states_list = [] 
        for i in range(self.lstm_step_count):
            h = torch.zeros((batch*topk, self.embed_dim), device=DEVICE)
            c = torch.zeros((batch*topk, self.embed_dim), device=DEVICE)
            hidden_states_list.append((h, c))

        # 3. attention context 초기화
        attn_context = torch.zeros((batch*topk, self.embed_dim), device = DEVICE)
        
        # 4. 초기 값 세팅
        ## input
        input = torch.full((batch*topk, ), VOCAB_MAP['<sos>'], device=DEVICE)
        ## 결과 저장용 
        predictions = torch.full((batch, topk, self.max_timesteps), VOCAB_MAP['<sos>'], dtype=torch.long, device=DEVICE)
        ## 종료 마스크
        finished = torch.zeros((batch, topk), dtype=torch.bool, device=DEVICE)
        
        # 5. log_probability는 batch 정상화 후 update (b, k)
            ## log prob이므로 0이 아니라 -무한대로 초기화(0이 가장 큰 수임.)
        beam_score = torch.full((batch, topk), -1e9, device=DEVICE)
        beam_score[:, 0] = 0.0  # 첫번째 log beam score = 0(무조건 선택)

        timesteps = self.max_timesteps

        # 6. decoding loop
        for t in range(timesteps-1):
            # beam search는 애초에 계속해서 스스로 선택한 답을 이어나가야 하므로 tf가 없음.
            # 1) input embedding
            char_embed = self.embedding(input) 

            # 2) attention context와 embedded input을 concat
            lstm_input = torch.cat((char_embed, attn_context), dim = -1)

            # 3) lstm_step에 통과 시키고, 꺼낸 hidden_state로 hidden_states_list를 갱신
            new_hidden = self.lstm_step(lstm_input, hidden_states_list) 
            hidden_states_list = new_hidden

            # 4) 가장 마지막 layer의 hidden state를 이용해서 attention context 계산
            last_hidden = hidden_states_list[-1][0]     # 가장 마지막 층의 hidden_state
            attn_context, attn_weights = self.attend.compute_context(last_hidden, encoder_len) 

            # 5) CDN 층에 통과
            cdn_input = torch.cat((last_hidden, attn_context), dim = -1) 
            raw_pred = self.CDN(cdn_input)                      # (b*k, vocab_size=31)
            log_prob = torch.log_softmax(raw_pred, dim = -1)    # (b*k, vocab_size=31)

            # 6) !!! <eos>를 뱉었던 beam은 <eos>만 생성하도록 만들기(masking) !!! 
            log_prob = log_prob.view(batch, topk, -1)   # (b, k, vocab_size)
            
            ## finished_mask 차원 변경 및 expand
                ### expand는 repeat처럼 비어있는 차원 쪽으로 복붙해서 늘려줌
                ### 단, 복사없이 view처럼 참조만 함 
            flat_mask = finished.unsqueeze(-1)                           # (b, k, 1)
            flat_mask_expand = flat_mask.expand(-1, -1, vocab_size)      # (b, k, 31)

            ## log_prob의 flat_mask부분은 모두 -1e9로 채우고, <eos>만 0으로 만들어 선택하도록 함.
            log_prob = log_prob.masked_fill(flat_mask_expand, -1e9)    
            log_prob[:, :, VOCAB_MAP['<eos>']] = log_prob[:, :, VOCAB_MAP['<eos>']].masked_fill(flat_mask, 0.0)

            # 7) beam score = 모든 확률의 곱(여기서는 log이므로 합으로 구함)
            updated_score = beam_score.unsqueeze(-1) + log_prob.view(batch, topk, -1) # (b, k, vocab_size = 31)
            
            ## 배치마다 beam_0의 31개 확률, beam_1의 31개 확률, ..., beam_k의 31개 확률 
            ## 이 나오도록 나열한 다음, 이중 topk만큼 선택
            updated_score = updated_score.view(batch, -1)   # (b, k*vocab_size)
            top_score, top_indices = torch.topk(updated_score, topk, dim=-1)    # 둘 모두 (b, k)

            ## beam_score update
            beam_score = top_score # (b, k)
            
            # 8) attn_context와 hidden_states를 위의 b*k로 합친 슈퍼배치에서 골라내기 위해 indexing
            ## batch
                ### 0-2, 3-5, ... 이렇게 3개씩이 하나의 배치를 의미
                ### batch_idx는 배치의 시작번호를 의미하므로, 이걸 beam_idx에 더하면
                ### b*k 중 어디가 우리가 원하는 건지 알아낼 수 있음.
            batch_idx = torch.arange(0, batch*topk, topk).to(DEVICE)    # (b, )
            
            ## top_indices를 이용해 한 배치 내의 topk개 서열 중 몇번째가 가장 높은지 확인
            beam_idx = top_indices // vocab_size    # (b, k)
            
            ## top_indices를 이용해 해당 서열이 몇번째 글자를 골랐는지 확인
            vocab_idx = top_indices % vocab_size    # (b, k)

            # 9) broadcasting summation으로 beam_idx에 batch 시작 번호를 더해 원하는 index 찾아내기
            select_idx = (batch_idx.unsqueeze(1) + beam_idx).view(-1)
            
            # 10) batch마다 topk만큼의 attn_context, hidden_states 추출
            attn_context = attn_context[select_idx]
            
            new_hidden_list = []
            for h, c in hidden_states_list:
                new_hidden_list.append((h[select_idx], c[select_idx]))
            hidden_states_list = new_hidden_list

            # 11) prediction sequence 업데이트
            predictions = predictions.view(batch*topk, -1)
            predictions = predictions[select_idx]
            predictions = predictions.view(batch, topk, -1)
            ## predictions에 새로운 토큰 추가
            predictions[:, :, t+1] = vocab_idx

            # 12) finished mask 업데이트
            ## select_idx인 beam만 선택
            finished = finished.view(-1)[select_idx]
            finished = finished.view(batch, topk)       # (b, k)

            ## vocab idx에 <eos>인 부분은 finished에 True
                ### | 는 bitwise 연산자로, 같은 shape의 boolean tensor에서
                ### 둘 중 하나라도 True이면 True, 둘 다 False이면 False인 tensor반환
            finished = finished | (vocab_idx == VOCAB_MAP['<eos>'])

            # 13)  다음 단계에 넣을 input 업데이트
            input = vocab_idx.view(-1)

        # 7. best predictions 반환
        best_predictions = predictions[:, 0, :]

        return best_predictions