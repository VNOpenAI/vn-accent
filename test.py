import argparse
import json
import os

import numpy as np
import torch
from tqdm import trange

from accent_utils import process_line
from utils import translate

from models.model_factory import get_model


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_path')
    parser.add_argument('weight_file')
    parser.add_argument('--test_data_file', default="data/test.notone")
    parser.add_argument('--ground_truth_file', default="data/test.tone")
    parser.add_argument('--config_file', default='model_config.json')
    parser.add_argument('--model_name', default='big_evolved')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()
    return args

def evaluate(pred, label):
    _, pred_punc = process_line(pred)
    _, label_punc = process_line(label)

    pred_punc = np.array(pred_punc)
    label_punc = np.array(label_punc)

    true_values = np.sum(pred_punc==label_punc)
    n_values = len(pred_punc)

    return true_values, n_values

if __name__=='__main__':
    args = get_arg()

    # Load tokenizer
    print("Load tokenizer")
    tokenizer = torch.load(args.vocab_path)
    src_tokenizer = tokenizer['notone']
    trg_tokenizer = tokenizer['tone']
    src_pad_token = 0
    trg_pad_token = 0

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

    model = get_model(model_param)
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


    test_data_lines = None
    ground_truth_lines = None
    with open(args.test_data_file, "r", encoding='utf-8') as f:
        test_data_lines = f.readlines()
    with open(args.ground_truth_file, "r", encoding='utf-8') as f:
        ground_truth_lines = f.readlines()

    total_true_values = 0
    total_values = 0
    t = trange(len(test_data_lines), desc='', leave=True)
    for i in t:
        line = test_data_lines[i]
        line_gt = ground_truth_lines[i]
        line_pr = translate(model, line, src_tokenizer, trg_tokenizer, use_mask=model_param["use_mask"], device=device)
        true_values, n_values = evaluate(line_pr, line_gt)
        total_true_values += true_values
        total_values += n_values
        t.set_description("Accuracy: {:.4f}".format(total_true_values / total_values))
        t.refresh() # to show immediately the update

    print("Avg. Accuracy: {}".format(total_true_values / total_values))
