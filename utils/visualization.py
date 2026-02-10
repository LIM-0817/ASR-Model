import torch
from torchviz import make_dot
from models import ASRModel
from config import DEVICE

model = ASRModel(batch_size=2, input_size=28, embed_dim=128, lstm_step=2).to(DEVICE)

# 가짜 데이터 생성 
dummy_x = torch.randn(2, 100, 28).to(DEVICE)    # (batch, seq, dim)
dummy_lx = torch.tensor([100, 100]).to(DEVICE)  # (batch, )
dummy_y = torch.zeros(2, 20).long().to(DEVICE)  # (batch, seq)

# feed forward
predictions, _ = model(dummy_x, dummy_lx, dummy_y, 1.0)

# 그래프 생성
## make_dot함수는 모델 출력값에서 시작해서 역전파 경로를 추적해 노드 생성
dot = make_dot(predictions, params=dict(model.named_parameters())) 
dot.format = 'png' # 또는 'pdf'
dot.render("LIM_ASR_Architecture_Graph")
print("아키텍쳐 시각화 자료 생성 완료")