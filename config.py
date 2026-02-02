import torch
import yaml

# 1. TOKEN 정의(alphabet)
VOCAB = [
    '<pad>', '<sos>', '<eos>',
    'A',   'B',    'C',    'D',
    'E',   'F',    'G',    'H',
    'I',   'J',    'K',    'L',
    'M',   'N',    'O',    'P',
    'Q',   'R',    'S',    'T',
    'U',   'V',    'W',    'X',
    'Y',   'Z',    "'",    ' ',
]
VOCAB_MAP = {VOCAB[i]:i for i in range(0, len(VOCAB))}

SOS_TOKEN = VOCAB_MAP['<sos>']
EOS_TOKEN = VOCAB_MAP['<eos>']
PAD_TOKEN = VOCAB_MAP['<pad>']


# 2. DEVICE 정의
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# 3. config dict - config.yaml을 dict로
def load_config(path):
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return config_dict

## config['name']으로 hyperparameter들에 접근 가능
config = load_config('config.yaml')