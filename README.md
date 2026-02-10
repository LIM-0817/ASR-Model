# End-to-End Automatic Speech Recognition (ASR) Model

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-1.13%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/WandB-Experiment-orange?style=for-the-badge&logo=weightsandbiases&logoColor=black"/>
</div>

<br>

**Listen, Attend and Spell (LAS)** 아키텍처를 기반
Carnegie Mellon Univ. (CMU) 11-785 Deep Learning 강좌의 HW4P2 구조를 시작으로 다양한 기법으로 성능을 끌어올림.

## Performance Improvement

수십 번의 실험과 구조 개선을 통해 초기 모델 대비 성능을 크게 향상시켰습니다.

| Decoding Strategy | Metric (Levenshtein Distance) | Improvement |
| :--- | :---: | :--- |
| **Baseline (Greedy)** | 23.xx | - |
| **Final (Beam Search)** | **18.xx (Public) / 21.xx (Private)** | **▼ Performance Boost** |

## Key Improvements 

베이스라인 모델의 한계를 극복하기 위해 다음과 같은 기법들을 단계적으로 적용했습니다.

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
| 학습 초기: 정렬이 형성되지 않음 | **학습 완료: 선명한 대각선(Diagonal) 형태의<br>Alignment가 형성됨을 확인** |

### 2. Training Log and(Wandb)
**Best Model 훈련 로그**
<img width="100%" alt="Best Model Log" src="https://github.com/user-attachments/assets/40c80e61-a906-4865-aa04-7c70f7a518f3" />

<details>
<summary><strong>📂 Click to see All Experiments History (Hyperparameter Tuning)</strong></summary>
<br>
모든 run
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