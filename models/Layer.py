import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UtilLayer import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        residual = x
        x = self.norm_1(x)
        x = self.attn(x,x,x,mask)
        x = self.dropout_1(x)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.dropout_2(x)
        x = residual + x
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        residual = x
        x = self.norm_1(x)
        x = self.attn_1(x,x,x,trg_mask)
        x = self.dropout_1(x)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.attn_2(x,e_outputs,e_outputs,src_mask)
        x = self.dropout_2(x)
        x = residual + x

        residual = x
        x = self.norm_3(x)
        x = self.ff(x)
        x = self.dropout_3(x)
        x = residual + x

        return x


