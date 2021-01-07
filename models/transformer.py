import torch.nn as nn

from .transformer_utils.models import Decoder, Encoder
from .transformer_utils import basic_layer


class TransformerEncoder(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, 
            d_model, d_ff, num_layers, num_heads, dropout, 
            layer_type=basic_layer):
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
            layer_type=basic_layer):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, num_heads, dropout, layer_type.EncoderLayer)
        self.decoder = Decoder(trg_vocab_size, d_model, d_ff, num_layers, num_heads, dropout, layer_type.DecoderLayer)
        self.out = nn.Linear(d_model, trg_vocab_size)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output
