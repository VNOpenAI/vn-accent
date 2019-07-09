import numpy as np
import torch
from torch.utils import data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class MTDataset(data.Dataset):
    def __init__(self, src_tokenizer, trg_tokenizer, src_corpus_path, trg_corpus_path, pad_len=200):
        with open(src_corpus_path) as f:
            src_data = f.read().split('\n')[:-1]
        with open(trg_corpus_path) as f:
            trg_data = f.read().split('\n')[:-1]
        
        src_tokens = src_tokenizer.texts_to_sequences(src_data)
        trg_tokens = trg_tokenizer.texts_to_sequences(trg_data)

        src_tokens = pad_sequences(src_tokens, pad_len)
        trg_tokens = pad_sequences(trg_tokens, pad_len)

        self.src = src_tokens.astype(np.int64)
        self.trg = trg_tokens.astype(np.int64)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, ix):
        return self.src[ix], self.trg[ix]