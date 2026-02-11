# End-to-End Automatic Speech Recognition (ASR) Model, (from Scratch)

**Listen, Attend and Spell (LAS)** 모델 참고, 처음부터 모든 구조 스스로 구현
- (Carnegie Mellon Univ. (CMU) 11-785 Deep Learning 강좌 HW4P2 참고)
- https://www.kaggle.com/competitions/attention-based-speech-recognition





## Model Architecture
```mermaid
graph TD
    %% 1. Input Section
    subgraph Input_Stage [Input Stage]
        In[Speech / Mfcc] --> Mel[Mel-Spectrogram]
    end

    %% 2. Listener Section (pBLSTM 반영)
    subgraph Listener [Listener: Encoder]
        Mel --> Conv[Conv1d / Stride]
        Conv --> pB1[pBLSTM Layer 1]
        pB1 --> pB2[pBLSTM Layer 2]
        pB2 --> pB3[pBLSTM Layer 3]
        note1[Each pBLSTM reduces<br/>Time-resolution by 1/2]
        pB1 -.-> note1
    end

    %% 3. Attender Section
    subgraph Attender [Attender]
        pB3 --> KeyValue[set_key_value:<br/>Generate Keys & Values]
        KeyValue --> Compute[compute_context:<br/>Dot-product Attention]
    end

    %% 4. Speller Section
    subgraph Speller [Speller: Decoder]
        SOS[< sos >] --> LSTM[LSTM Cells]
        LSTM --> Query[Generate Query Qk]
        Query --> Compute
        Compute --> Context[Attention Context Vector]
        Context --> CDN[Character Distribution Network]
    end

    %% 5. Inference Strategy (Beam Search 반영)
    subgraph Inference [Inference Strategy]
        CDN --> Greedy[Greedy Search]
        CDN --> Beam[Beam Search: Top-K Hypotheses]
        style Beam fill:#f96,stroke:#333,stroke-width:2px
    end

    %% 6. Final Result
    Beam --> Final[Final Result: 18.3 CER]
    Greedy --> Base[Baseline: 23.9 CER]

    %% Styling
    style Listener fill:#fff4dd,stroke:#d4a017
    style Attender fill:#e1f5fe,stroke:#01579b
    style Speller fill:#e8f5e9,stroke:#2e7d32
    style Final fill:#00c853,stroke:#333,stroke-width:4p
```
</div> <p><b>▲ Encoder-Attender-Decoder을 거치는 end-to-end 구조</b></p> </div>


<img width="412" height="161" alt="Image" src="https://github.com/user-attachments/assets/b0fe6a13-74c8-4860-bc53-02d88eba39af" />
</div> <p><b>▲ Listener 내부 구조</b></p> </div>


<img width="553" height="384" alt="Image" src="https://github.com/user-attachments/assets/a6533952-4463-4e01-abac-8837867e9b55" />
</div> <p><b>▲ speller 내부 구조</b></p> </div>


## Best model
Validation Levenshtein Distance: 23.926

Validation Loss: 0.6045

Test Levenshtein Distance (Public): 18.3597 (Beam Search 적용 결과)

<div align="center"> <div style="border: 2px solid #e1e4e8; border-radius: 10px; padding: 10px; display: inline-block; background-color: #f6f8fa;"> <img width="1101" alt="Kaggle Submission Result" src="https://github.com/user-attachments/assets/d219963d-180a-444a-8c10-d4d12a2daede" /> </div> <p><b>▲ Kaggle 제출 및 스코어 확인 화면</b></p> </div>




## Key Improvements 


### 1. Architecture Enhancements
- **PBLSTM (Pyramidal Bi-LSTM)**: 시간 차원을 압축하여 긴 시퀀스 학습 효율 증대
- **Add one more layer of PBLSTM**: 인코더의 깊이를 늘려 음성 특징 추출 능력 강화
- **Conv1d Stride Tuning**


### 2. Training Strategy
- **Scaling Factor**: 그래디언트 소실폭발 방지
- **Staged Teacher Forcing Ratio**: 2 staged teacher forcing decay를 이용
- **Spec Augmentation**: Time Masking, Frequency Masking을 통한 데이터 증강


### 3. Inference & Attention
- **Attention Padding Masking**
- **Beam Search Implementation**: 단순 Greedy Decoding 대신 test시 beam search이용해 레벤슈타인 거리 감소




## Visualization & Analysis


### 1. Attention Map Analysis

| Epoch 1 (Initial) | Epoch 150 (Converged) |
| :---: | :---: |
| <img width="100%" src="https://github.com/user-attachments/assets/fe2a73ce-19a4-40ee-9003-cc01b9e38298" /> | <img width="100%" src="https://github.com/user-attachments/assets/ed7ccb1d-269b-4ecf-9b0e-e1976f29300c" /> |
| 학습 초기: 정렬이 형성되지 않음 | **학습 완료: 약간 대각선(Diagonal) 형태의<br>Alignment가 형성** |

**단**, LAS 기반의 구조에서는 완전히 attention자체에 의존하지 않으므로 대각선이 선명하게 생기기는 어려운 측면이 있음. 




### 2. Training Log and(Wandb)
**Best Model 훈련 로그**
<img width="100%" alt="Best Model Log" src="https://github.com/user-attachments/assets/40c80e61-a906-4865-aa04-7c70f7a518f3" />


<details>
<summary><strong> 모든 run 확인하고 싶은 경우 클릭</strong></summary>
<br>
RUNS
<br><br>
<img width="100%" alt="All Run Log" src="https://github.com/user-attachments/assets/f80bc842-c170-480d-817a-67204283a658" />
</details>




## Installation & Usage


### 1. Requirements
```bash
pip install -r requirements.txt
```




### 2. dataset download
!! kaggle 계정에서 API 생성 후 .kaggle 폴더에 업로드 필요 !!

```bash
# 1. Kaggle API 설치
pip install -q kaggle

# 2. 데이터셋 다운로드
kaggle competitions download -c attention-based-speech-recognition -p ./data

# 3. 압축 해제
unzip -q ./data/attention-based-speech-recognition.zip -d ./data
```




### 3. Train
config.yaml 파일에서 hyperparameter tuning 이후 학습을 진행(config.py는 건드리지 말기!)
```bash
python train.py
```




### 4. Test(Inference)
```bash
python test.py
```




## Project Structure
```
.
├── models/             # Listener, Speller, Attention modules
├── utils/              # Helpers, Metrics, Visualization
├── config.yaml         # Hyperparameter configuration
├── train.py            # Training script
├── test.py             # Inference script (Beam Search included)
├── dataset.py
└── README.md
```