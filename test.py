import os
import torch
import pandas as pd
from tqdm import tqdm as blue_tqdm

from utils.metrics import indices_to_chars
from config import VOCAB, DEVICE, config
from dataset import SpeechDatasetTest
from models import ASRModel


DATA_DIR        = 'data/11-785-f23-hw4p2'
PARTITION       = config['train_dataset']
CEPSTRAL        = config['cepstral_norm']


test_dataset    = SpeechDatasetTest(
    root        = DATA_DIR,
    partition   = 'test-clean',
    cepstral    = CEPSTRAL,
)

test_loader     = torch.utils.data.DataLoader(
    dataset     = test_dataset,
    batch_size  = config['batch_size'],
    shuffle     = False, #Test 시에도 shuffle을 꺼도 됨.
    num_workers = config['num_workers'],
    pin_memory  = True,
    collate_fn  = test_dataset.collate_fn,
    drop_last   = False
)

# 1. model 정의(ASR model)
model = ASRModel(
    listener_hidden_size=config["listener_hidden_size"], 
    batch_size = config['batch_size'], 
    input_size = 28, 
    embed_dim = config["embed_dim"],  
    lstm_step = 2,
    decode_mode = config["decode_mode"]
    )

model = model.to(DEVICE)

def test(model, dataloader):
    model.eval()
    batch_bar = blue_tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Test")

    predictions = []

    for i, (x, lx) in enumerate(dataloader):
        x, lx = x.to(DEVICE), lx.to(DEVICE)

        with torch.inference_mode():
            predictions, _ = model(x, lx, y = None, decode_mode=config["decode_mode"])

        # 2. Greedy Decoding
        if model.decode_mode == "greedy":
            greedy_predictions = torch.argmax(predictions, dim = -1)

        ## Convert predictions to characters
            for pred in greedy_predictions:
                predictions.append("".join(indices_to_chars(pred, VOCAB)))
        
        # 1. beam search
        ## beam search는 변환없이 바로 indices_to_chars 적용
        if model.decode_mode == "beam":
            for pred in greedy_predictions:
                    predictions.append("".join(indices_to_chars(pred, VOCAB)))

        batch_bar.update()

        del x, lx
        torch.cuda.empty_cache()

    batch_bar.close()
    return predictions



def main():
    # best model을 불러옴
    checkpoint_path = './MyModelCheckpoints/best_model.pth'
    
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model checkpoint loaded successfully!")

    except FileNotFoundError:
        print("Best model checkpoint not found. Please ensure 'best_model.pth' exists.")

    # testing
    test_predictions = test(model, test_loader)
    submission_df = pd.DataFrame({'index': range(len(test_predictions)), 'label': test_predictions})

    # kaggle submission file(.csv)만들기 
    submission_df.to_csv('submission.csv', index=False)

    print("Submission file 'submission.csv' created successfully!")


if __name__ == "__main__":
    main()