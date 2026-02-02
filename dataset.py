import os
import torch
import numpy as np
from tqdm import tqdm as blue_tqdm

from config import VOCAB, VOCAB_MAP


# Memory efficient, 훈련에 사용
class SpeechDatasetME(torch.utils.data.Dataset): 
    def __init__(self, root, partition= "train-clean-360", cepstral=True):
        self.VOCAB = VOCAB
        self.cepstral = cepstral

        # partition이 입력된 경우 해당 partition만 이용해서 사용
        if partition == "train-clean-100" or partition == "train-clean-360" or partition =="dev-clean":
            mfcc_dir = f'./{root}/{partition}/mfcc'                     # mfcc 경로
            transcript_dir = f'./{root}/{partition}/transcripts'        # transcript 경로

            # mfcc 폴더 내의 모든 mfcc 파일 경로 list
            mfcc_files = [] 
            for file in os.listdir(mfcc_dir):
                mfcc_files.append(os.path.join(mfcc_dir, file))
            
            # append시 순서가 다를 수 있으니 이름 순으로 sort
            mfcc_files = sorted(mfcc_files)                             

            # 마찬가지로 transcript 파일 경로 정리
            transcript_files = [] 
            for file in os.listdir(transcript_dir):
                transcript_files.append(os.path.join(transcript_dir, file))

            # append시 순서가 다를 수 있으니 이름 순으로 sort
            transcript_files = sorted(transcript_files) 

        # partition이 정해진게 아니라면 100과 360 모두를 training에 사용
        else: 
            mfcc_dir = f'./{root}/train-clean-100/mfcc'          
            transcript_dir = f'./{root}/train-clean-100/transcripts'   

            mfcc_files = [] 
            for file in os.listdir(mfcc_dir):
                mfcc_files.append(os.path.join(mfcc_dir, file))

            transcript_files = []
            for file in os.listdir(transcript_dir):
                transcript_files.append(os.path.join(transcript_dir, file))

            mfcc_dir = f'./{root}/train-clean-360/mfcc' 
            transcript_dir = f'./{root}/train-clean-360/transcripts' 

            # 모든 파일 경로들을 list에 append
            for file in os.listdir(mfcc_dir): 
                mfcc_files.append(os.path.join(mfcc_dir, file))
            mfcc_files = sorted(mfcc_files) 

            for file in os.listdir(transcript_dir): 
                transcript_files.append(os.path.join(transcript_dir, file))
            transcript_files = sorted(transcript_files) 

        assert len(mfcc_files) == len(transcript_files)

        self.mfcc_files = mfcc_files
        self.transcript_files = transcript_files
        self.length = len(transcript_files)
        print("Loaded file paths ME: ", partition)

    # dataset len 함수 정의
    def __len__(self):
        return self.length

    # dataset 데이터 접근 함수 getitem 정의
    def __getitem__(self, ind):
        # mfcc, transcript 파일 load
        mfcc = np.load(self.mfcc_files[ind]) 
        transcript = np.load(self.transcript_files[ind]) 

        # mfcc 정규화
        mfcc_mean = mfcc.mean(axis = -1, keepdims= True)
        mfcc_std = mfcc.std(axis = -1, keepdims= True)
        # 주의!!! numpy 계산 시 여기서 mean을 쓰면 (T, D)와 (T, )를 연산해야되는데, broadcast가 불가능함
        # 따라서 keepdims = True를 통해 2차원의 dim을 유지해 (T, 1)로 만들어줘야하는 것.
        mfcc = (mfcc - mfcc_mean)/mfcc_std

        #list comprehension을 이용해서 transcript를 숫자 sequence로 변경
        transcript_mapped = [VOCAB_MAP[i] for i in transcript]

        return torch.FloatTensor(mfcc), torch.LongTensor(transcript_mapped)

    def collate_fn(self, batch):
    # !dataloader은 __getitem__을 배치 size 만큼 호출해 batch=[]안에 담음. 그리고 이 batch를 collate_fn에 마지막에 넣어줌!
    # 현재 dataset의 __getitem__에서 tuple을 반환. 따라서 dataloader에서 batch_size가 4라면, [(mfcc1, trans1), (mfcc2, trans2), (mfcc3, trans3), (mfcc4, trans4)]가 한 배치가 됨
    # 이걸 collate_fn의 input인 'batch'에 넣어주는 것.

        batch_x, batch_y, lengths_x, lengths_y = [], [], [], []

        # mfcc, transcript와 len들을 각 list에 append
        for x, y in batch:
            batch_x.append(x)
            lengths_x.append(x.shape[0])
            batch_y.append(y)
            lengths_y.append(y.shape[0])

        # padding
        batch_x_pad = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first = True)
        batch_y_pad = torch.nn.utils.rnn.pad_sequence(batch_y, batch_first = True)

        return batch_x_pad, batch_y_pad, torch.tensor(lengths_x), torch.tensor(lengths_y)
    

# Test에 사용할 Dataset class
class SpeechDatasetTest(torch.utils.data.Dataset):  
    def __init__(self, root, partition, cepstral=False):
        self.mfcc_dir = f'./{root}/{partition}/mfcc'            # test-clean mfccs 경로
        self.mfcc_files = sorted(os.listdir(self.mfcc_dir))     # mfcc 폴더 파일 리스트

        self.mfccs = []
        for i, filename in enumerate(blue_tqdm(self.mfcc_files)):
            mfcc = np.load(os.path.join(self.mfcc_dir, filename)) # mfcc 파일 load
            if cepstral:
                # mfcc 값 정규화
                mfcc = (mfcc - mfcc.mean(axis = -1, keepdims= True))/mfcc.std(axis = -1, keepdims= True)
            # 리스트에 append
            self.mfccs.append(mfcc)

        print("Loaded: ", partition)

    def __len__(self):
        return len(self.mfccs)

    def __getitem__(self, ind):
        mfcc_data = self.mfccs[ind]
        return torch.from_numpy(mfcc_data)

    def collate_fn(self,batch):
        batch_x, lengths_x = [], []
        for x in batch:
            # Append the mfccs and their lengths to the lists created above
            batch_x.append(x)
            lengths_x.append(x.shape[0])

        # padding
        batch_x_pad = torch.nn.utils.rnn.pad_sequence(batch_x, batch_first = True) 
        return batch_x_pad, torch.tensor(lengths_x)