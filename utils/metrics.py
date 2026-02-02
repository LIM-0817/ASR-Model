# 해당 코드는 HW4P2-Attention based speech recognition.ipynb의 'Levenstein Distance'을 그대로 가져옴.

import Levenshtein
from config import *

# We have given you this utility function which takes a sequence of indices and converts them to a list of characters
def indices_to_chars(indices, vocab):
    tokens = []
    for i in indices: # This loops through all the indices
        if int(i) == SOS_TOKEN: # If SOS is encountered, dont add it to the final list
            continue
        elif int(i) == EOS_TOKEN: # If EOS is encountered, stop the decoding process
            break
        else:
            tokens.append(vocab[i])
    return tokens

# To make your life more easier, we have given the Levenshtein distantce / Edit distance calculation code
def calc_edit_distance(predictions, y, y_len, vocab= VOCAB, print_example= False):

    dist                = 0
    batch_size, seq_len = predictions.shape
    EOS_TOKEN_INDEX = 2

    for batch_idx in range(batch_size):

        y_sliced    = indices_to_chars(y[batch_idx,0:y_len[batch_idx]], vocab)

        #첫번째 EOS 토큰 index찾기
        end_indices = (predictions[batch_idx] == EOS_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        
        if len(end_indices) > 0:
            end_index = end_indices[0].item()
        else:
            # EOS 토큰이 없으면 예측 텐서 전체를 사용
            end_index = predictions[batch_idx].shape[0]        
        
        pred_sliced = indices_to_chars(predictions[batch_idx], vocab)

        # Strings - When you are using characters from the AudioDataset
        y_string    = ''.join(y_sliced)
        pred_string = ''.join(pred_sliced)

        #dist        += Levenshtein.distance(pred_string, y_string)
        # Comment the above abd uncomment below for toy dataset
        dist      += Levenshtein.distance(y_sliced, pred_sliced)

    if print_example:
        # Print y_sliced and pred_sliced if you are using the toy dataset
        print("\nGround Truth : ", y_string)
        print("Prediction   : ", pred_string)

    dist    /= batch_size
    return dist