import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from model import get_model
from utils import translate
from tqdm import tqdm

def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_path')
    parser.add_argument('test_path')
    parser.add_argument('weight_file')
    parser.add_argument('--output_file', default='output.txt')
    parser.add_argument('--config_file', default='model_config.json')
    parser.add_argument('--model_name', default='big_evolved')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)

    args = parser.parse_args()

    return args

def make_dataset(fp, tokenizer, max_len=200):
    with open(fp) as f:
        sents = f.read().split('\n')[:-1]

    seqs = tokenizer.texts_to_sequences(sents)
    seqs_len = [len(x) for x in seqs]
    seqs = pad_sequences(seqs, max_len)
    seq_tensors = torch.tensor(seqs).long()
    seq_len_tensors = torch.tensor(seqs_len).long()
    ds = data.dataset.TensorDataset(seq_tensors, seq_len_tensors)
    return ds


if __name__=='__main__':
    args = get_arg()

    # Load tokenizer
    print("Load tokenizer")
    tokenizer = torch.load(args.vocab_path)
    src_tokenizer = tokenizer['notone']
    trg_tokenizer = tokenizer['tone']
    src_pad_token = 0
    trg_pad_token = 0

    # Load data
    print("Load data")
    dataset = make_dataset(args.test_path, src_tokenizer)
    data_iter = data.dataloader.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    print("Init model")
    with open(args.config_file) as f:
        config = json.load(f)
    
    if args.model_name in config:
        model_param = config[args.model_name]
    else:
        raise Exception("Invalid model name")
    
    model_param['src_vocab_size'] = len(src_tokenizer.word_index) + 1
    model_param['trg_vocab_size'] = len(trg_tokenizer.word_index) + 1

    model = get_model(**model_param)
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    print("Using", device.type)
    if device.type=='cuda':
        model = model.cuda()

    if os.path.isfile(args.weight_file):
        print("Load model")
        state = torch.load(args.weight_file)
        if isinstance(state, dict):
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
    else:
        raise Exception("Invalid weight path")

    with tqdm(total=len(data_iter)) as pbar, open(args.output_file, 'w') as f:
        for seqs, seqs_len in data_iter:
            if device.type=='cuda':
                seqs = seqs.cuda()

            with torch.no_grad():
                logit = forward(model, seqs, src_pad_token)
            logit = F.softmax(logit, dim=-1)
            preds = logit.argmax(dim=-1)
            preds = preds.cpu().numpy()
            seqs_len = seqs_len.numpy()
            trg_seqs = []
            for seq, seq_len in zip(preds, seqs_len):
                seq = seq[:seq_len]
                trg_seqs.append(seq)

            trg_sents = trg_tokenizer.sequences_to_texts(trg_seqs)
            for s in trg_sents:
                f.write(s+'\n')
            pbar.update(1)


