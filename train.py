import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm as blue_tqdm
from torch.amp import GradScaler, autocast
import wandb

from config import VOCAB, DEVICE, config
from models import ASRModel
from utils.metrics import calc_edit_distance
from utils.helpers import plot_attention, TimeElapsed
from dataset import SpeechDatasetME


# 1. model 정의(ASR model)
model = ASRModel(batch_size = config['batch_size'], input_size = 28, embed_dim =128,  lstm_step = 2)
model = model.to(DEVICE)


# 2. optimizer, loss fx, scaler, scheduler 정의
optimizer   = torch.optim.Adam(model.parameters(), lr = config['learning_rate'])
criterion   = nn.CrossEntropyLoss(ignore_index = 0)

## scaler: 
## - autocast를 이용하는 경우 FP32(32 bit floating point)를 자동으로 FP16으로 바꿔서 사용하여 연산 효율화(vram 절약)
## - 그런데, 이 과정에서 loss값이 너무 작아 FP16의 메모리 범위를 넘어가는 경우 그냥 0이 되어버림
## - 따라서 scaler은 2^16등의 큰 값을 loss에 곱한 후(scale) 
## - 역전파를 수행한 뒤
## - 다시 2^16을 나누어 원래 크기로 바꾸고,
## - 이를 반영하여 optimizer이 가중치를 업데이트하는 방식을 사용함. 
scaler      = GradScaler()

## scheduler:
## 성능 개선이 더딘 경우, lr이 너무 높아서 그렇다고 판단하고 lr을 낮줘줌.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",        # loss 감소를 모니터링할 때
    factor=0.5,        # lr을 절반으로 줄임
    patience=3,        # 3 epoch 동안 개선 없으면 감소
    threshold=1e-3,    # 개선으로 인정할 최소 변화
    min_lr=1e-6        # lr이 너무 낮아지는 것 방지
)


# 3. train함수 정의
def train(model, dataloader, criterion, optimizer, teacher_forcing_ratio):

    model.train()
    batch_bar = blue_tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train')

    running_loss        = 0.0
    running_perplexity  = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):

        optimizer.zero_grad()

        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx.to(DEVICE), ly.to(DEVICE)

        with autocast(device_type=DEVICE): # autocast 실행 상태에서 loss 계산

            raw_predictions, attention_plot = model(x, lx, y= y, teacher_forcing_ratio = teacher_forcing_ratio)
            # nn.CrossEntropyLoss의 입력조건은 (N, C), (N,)임
            # 따라서 (batch_size, timesteps, vocab_size), (batch_size, timesteps)인 애들의 shape을 바꿔줘야 함.
            loss = criterion(raw_predictions[:, :-1, :].reshape(-1, raw_predictions.size(-1)), y[:, 1:].reshape(-1))
            perplexity = torch.exp(loss) # Perplexity is defined the exponential of the loss
            running_loss += loss.item()
            running_perplexity += perplexity.item()


        if not torch.isfinite(loss):
            print(f"Warning: Loss is not finite! Value: {loss}")
            # 디버깅: 텐서값이 혹시 무한대인지 확인

        # scale한 상태에서 역전파로 loss 계산 수행
        scaler.scale(loss).backward()

        # unscale 및 가중치 업데이트
        scaler.step(optimizer)

        # scaler 계수(loss에 곱해주는 값) 업데이트 - loss가 작을수록 큼
        scaler.update()


        batch_bar.set_postfix(
            loss="{:.04f}".format(running_loss/(i+1)),
            perplexity="{:.04f}".format(running_perplexity/(i+1)),
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr'])),
            tf_rate='{:.02f}'.format(teacher_forcing_ratio))
        batch_bar.update()

        del x, y, lx, ly
        torch.cuda.empty_cache()

    running_loss /= len(dataloader)
    running_perplexity /= len(dataloader)
    batch_bar.close()

    return running_loss, running_perplexity, attention_plot


# 4. valid함수 정의
def validate(model, dataloader):

    model.eval()

    batch_bar = blue_tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    running_lev_dist = 0.0

    for i, (x, y, lx, ly) in enumerate(dataloader):
        
        x, y, lx, ly = x.to(DEVICE), y.to(DEVICE), lx.to(DEVICE), ly.to(DEVICE)

        try:
            with torch.inference_mode():
                # 모델에 데이터 전달
                raw_predictions, attentions = model(x, lx, y=None)

            # Greedy Decoding
            greedy_predictions   = torch.argmax(raw_predictions, dim = -1) # TODO: How do you get the most likely character from each distribution in the batch?
    
            # Levenshtein Distance 계산
            running_lev_dist    += calc_edit_distance(greedy_predictions, y, ly, VOCAB, print_example = False) # You can use print_example = True for one specific index i in your batches if you want
    
            batch_bar.set_postfix(
                dist="{:.04f}".format(running_lev_dist/(i+1)))
            batch_bar.update()
    
            del x, y, lx, ly
            torch.cuda.empty_cache()
        
        except RuntimeError as e:
            # 에러가 발생해도 계속 진행
            print(f"\n[Validation Error] 배치를 건너뜁니다: {e}")
            batch_bar.set_description("Val (Error)") # 에러 상태를 표시

    batch_bar.close()
    running_lev_dist /= len(dataloader)

    return running_lev_dist



# 훈련을 진행할 main 함수 정의
def main():
    # 0. 메모리 정리
    torch.cuda.empty_cache()    # 파이썬 레벨에서 더 이상 사용하지 않는 객체를 찾아 메모리에서 해제
    gc.collect()                # 파이토치가 미리 잡아두고 있는 GPU 캐시 메모리를 강제로 OS에 반환

    # 1. dataset 불러오기
    DATA_DIR        = 'data/11-785-f23-hw4p2'
    PARTITION       = config['train_dataset']
    CEPSTRAL        = config['cepstral_norm']

    train_dataset   = SpeechDatasetME( # Or AudioDatasetME
        root        = DATA_DIR,
        partition   = PARTITION,
        cepstral    = CEPSTRAL
    )

    valid_dataset   = SpeechDatasetME(
        root        = DATA_DIR,
        partition   = 'dev-clean',
        cepstral    = CEPSTRAL
    )

    # 2. dataloader
    train_loader    = torch.utils.data.DataLoader(
        dataset     = train_dataset,
        batch_size  = config['batch_size'],
        shuffle     = True,
        num_workers = config['num_workers'],
        pin_memory  = True,
        collate_fn  = train_dataset.collate_fn, #이런 식으로 dataset에서 collate_fn을 정의한 뒤 train_loader에서 collate_fn을 불러옴.
        drop_last = True
    )

    valid_loader    = torch.utils.data.DataLoader(
        dataset     = valid_dataset,
        batch_size  = config['batch_size'],
        shuffle     = False, #validation 시에는 shuffle을 꺼도 됨.
        num_workers = config['num_workers'],
        pin_memory  = True,
        collate_fn  = valid_dataset.collate_fn,
        drop_last = True
    )

    
    # 3. wandb 설정
    wandb.login()

    ## project 이름과 run의 이름을 지정
    wandb.init(project = 'attention-based-speech-recognition-Lim', name = config['project_name']) 

    ## model = ASRmodel을 ASR_model.txt에 저장
    with open("ASR_model.txt", "w") as f:
        f.write(str(model))
    wandb.save("ASR_model.txt") # wandb에 저장

    # 4. 모델 저장할 경로 설정
    drive_checkpoint_dir = './MyModelCheckpoints'

    if not os.path.exists(drive_checkpoint_dir):
        os.makedirs(drive_checkpoint_dir)
        print(f"폴더를 생성했습니다: {drive_checkpoint_dir}")
    checkpoint_path = os.path.join(drive_checkpoint_dir, 'best_model.pth')


    # 5. Training
    ## 기본값 설정
    best_lev_dist = float("inf")
    tf_rate = config['tf_rate']
    tf_decay_factor = config['tf_decay_factor'] # 매 epoch마다 얼마나 tf 값을 줄일지
    timer = TimeElapsed()



    ## 만약에 best_model이 저장되어있고, config.yaml의 resume_training이 true라면 이어서 훈련
    resume_training = config['resume_training']
    if resume_training == True and os.path.exists(checkpoint_path):
        print("---훈련 재개---")
        checkpoint = torch.load(checkpoint_path)
        
        # 모델 가중치 복원
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Optimizer 및 Scheduler 상태 복원
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 학습률 강제 변경
        lr_input = config['learning_rate']
        current_lr = optimizer.param_groups[0]['lr']
        
        ## config.yaml의 lr과 현재 optimizer에 저장된 lr이 다르다면 config.yaml의 lr로 update
        if lr_input != current_lr:
            print(f'learning rate를 변경합니다.: {current_lr} -> {lr_input}')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_input

        ## scheduler에 기억되어있는 값들 변경
        ## 새 학습률로 미세 조정을 시작하므로, 기억하고 있는 valid dist와 patience 초기화 
        scheduler.best = float('inf')
        scheduler.num_bad_epochs = 0
        
        # 스케줄러가 마지막으로 인지한 학습률도 업데이트
        if hasattr(scheduler, '_last_lr'):
            scheduler._last_lr = [lr_input]

        # best_lev_dist 불러오기
        ## best_lev_dist가 있다면 해당 값을, 없다면 양의 무한대를 불러옴.
        best_lev_dist = checkpoint.get('best_lev_dist', float('inf'))
        
        # 시작 에폭 설정
        start_epoch = checkpoint['epoch'] + 1 
    else:
        start_epoch = 0
        print("--- 처음부터 훈련 시작 ---")

    
    ## 훈련 loop 시작
    for epoch in range(start_epoch, config['epochs']):
        ### 0) 타이머 시작 및 epoch 시작 log
        timer.time_elapsed()
        print("\nEpoch: {}/{}".format(epoch+1, config['epochs']))

        ### 1) train 함수 실행
        train_loss, perplexity, attention_plot = train(
            model,
            train_loader,
            criterion,
            optimizer,
            teacher_forcing_ratio = tf_rate,
        )
        
        ### 2) validate 함수 실행
        valid_dist = validate(
            model,
            valid_loader,
        )

        timer.time_elapsed() # timer 종료 및 진행 시간 print

        ### 3) Train Loss, valid_dist 로그
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Levenshtein Distance: {valid_dist:.4f}")

        ## 4) attention plotting
        plot_attention(attention_plot[0].cpu().detach().numpy(), epoch=epoch)

        ## 5) Log metrics to Wandb
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,                           # train_loss 로깅
            "valid_levenshtein_distance": valid_dist,           # valid_dist 로깅
            "learning_rate": optimizer.param_groups[0]['lr'],   # lr 로깅
            "tf_rate": tf_rate,
            "attention_plot": wandb.Image(
                attention_plot[0].cpu().detach().numpy(), 
                caption=f"Epoch {epoch+1} Attention"
                )
        }, step=epoch)

        ### 6) scheduler로 lr 조절
        if scheduler and epoch >= 20:
            scheduler.step(valid_dist) 

        ### 7) tf rate 낮추기
        tf_rate *= tf_decay_factor
        tf_rate = max(tf_rate, 0.6) # tf_rate 값 제한(0.6)

        ### 8) Levenshtein Distance가 낮게 나왔으면 best_model.pth 갱신
        if valid_dist < best_lev_dist: 
            print(f"New best model found! Levenshtein distance improved from {best_lev_dist:.4f} to {valid_dist:.4f}")
            best_lev_dist = valid_dist
            # Save your model checkpoint here
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_lev_dist': best_lev_dist,
            }, checkpoint_path) 

            print(f"모델 체크포인트 저장 완료: {checkpoint_path}")



if __name__ == "__main__":
    main()