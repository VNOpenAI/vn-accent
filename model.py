import os
import torch
import torch.nn as nn

from Transformer.Model import Layer, EvolvedLayer
from Transformer.Model.Transformer import Encoder, Decoder


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



def get_model(src_vocab_size, trg_vocab_size, 
        d_model=512, d_ff=2048, 
        num_layers=6, num_heads=8, dropout=0.1, 
        is_evolved=False, only_encoder=False):

    assert d_model % num_heads == 0
    assert dropout < 1

    model_type = None
    layer_type = None

    if is_evolved:
        layer_type = EvolvedLayer
    else:
        layer_type = Layer

    if only_encoder:
        model_type = TransformerEncoder
    else:
        model_type = Transformer

    model = model_type(src_vocab_size, trg_vocab_size, 
                                d_model, d_ff, num_layers, num_heads, 
                                dropout, layer_type=layer_type)

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