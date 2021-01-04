import os
import torch
import torch.nn as nn

from models import Layer, EvolvedLayer
from models.Transformer import Encoder, Decoder


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
            d_model, d_ff, num_layers, num_heads, dropout, 
            layer_type=Layer):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, num_heads, dropout, layer_type.EncoderLayer)
        self.out = nn.Linear(d_model, trg_vocab_size)
    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        output = self.out(e_outputs)
        return output


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
            d_model, d_ff, num_layers, num_heads, dropout, 
            layer_type=Layer):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, num_heads, dropout, layer_type.EncoderLayer)
        self.decoder = Decoder(trg_vocab_size, d_model, d_ff, num_layers, num_heads, dropout, layer_type.DecoderLayer)
        self.out = nn.Linear(d_model, trg_vocab_size)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output


class LSTM(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
            d_model, dropout):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, d_model)
        self.lstm = nn.LSTM(d_model, d_model)
        self.out = nn.Linear(d_model, trg_vocab_size)
    def forward(self, src):
        embedded = self.embed(src)
        lstm_output, hidden = self.lstm(embedded)
        output = self.out(lstm_output)
        return output


def get_model(model_type, src_vocab_size, trg_vocab_size, 
        d_model=512, d_ff=2048, 
        num_layers=6, num_heads=8, dropout=0.1):

    assert d_model % num_heads == 0
    assert dropout < 1


    model = None
    if model_type == "TRANSFORMER_BASE":
        model = Transformer(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=Layer)
    elif model_type == "TRANSFORMER_ENCODER_BASE":
        model = TransformerEncoder(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=Layer)
    elif model_type == "TRANSFORMER_EVOLVED":
        model = Transformer(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=EvolvedLayer)
    elif model_type == "TRANSFORMER_ENCODER_EVOLVED":
        model = TransformerEncoder(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=EvolvedLayer)
    elif model_type == "LSTM":
        model = LSTM(src_vocab_size, trg_vocab_size, 
                                d_model, 
                                dropout)

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