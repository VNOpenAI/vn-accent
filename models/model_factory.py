import os
import torch
import torch.nn as nn

from .transformer_utils import evolved_layer, basic_layer

from .transformer import TransformerEncoder, Transformer
from .lstm import LSTM


def get_model(model_param):
    model_type = model_param.get("model_type")
    src_vocab_size = model_param.get("src_vocab_size")
    trg_vocab_size = model_param.get("trg_vocab_size")
    d_model = model_param.get("d_model", 512)
    d_ff = model_param.get("d_ff", 2048)
    num_layers = model_param.get("num_layers", 6)
    num_heads = model_param.get("num_heads", 8)
    dropout = model_param.get("dropout", 0.0)

    # assert d_model % num_heads == 0
    assert dropout < 1

    model = None
    if model_type == "TRANSFORMER_BASE":
        model = Transformer(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=basic_layer)
    elif model_type == "TRANSFORMER_ENCODER_BASE":
        model = TransformerEncoder(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=basic_layer)
    elif model_type == "TRANSFORMER_EVOLVED":
        model = Transformer(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=evolved_layer)
    elif model_type == "TRANSFORMER_ENCODER_EVOLVED":
        model = TransformerEncoder(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=evolved_layer)
    elif model_type == "LSTM":
        model = LSTM(src_vocab_size, trg_vocab_size, d_model, bidirectional=False)
    elif model_type == "LSTM_BIDIRECTIONAL":
        model = LSTM(src_vocab_size, trg_vocab_size, d_model, bidirectional=True)
    else:
        raise ValueError("Wrong model type: {}".format(model_type))

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p) 

    return model

def load_model(model, optim=None, sched=None, path=''):
    if os.path.isfile(path):
        state = torch.load(path)
        model.load_state_dict(state['model'])
        if optim is not None:
            optim.load_state_dict(state['optim'])
        if sched is not None:
            sched.load_state_dict(state['sched'])
    else:
        raise Exception("Invalid path")