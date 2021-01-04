import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules.lightweight_convolution import LightweightConv1dTBC

from .UtilLayer import *


class LightweightConvLayer(nn.Module):
    def __init__(self, d_model, conv_dim, kernel_size, weight_softmax, num_heads, weight_dropout):
        super().__init__()
        padding_l = kernel_size // 2 if kernel_size % 2 == 1 else ((kernel_size - 1) // 2, kernel_size // 2)
        self.linear_1 = nn.Linear(d_model, conv_dim*2)
        self.activation = nn.GLU()
        self.conv = LightweightConv1dTBC(conv_dim, kernel_size, padding_l=padding_l,
                                            weight_softmax=weight_softmax,
                                            num_heads=num_heads,
                                            weight_dropout=weight_dropout)
        self.linear_2 = nn.Linear(conv_dim, d_model)

    def forward(self, x, mask):
        x = self.linear_1(x)
        x = self.activation(x)
        conv_mask = mask[:,-1,:] # BxS
        conv_mask = conv_mask.unsqueeze(-1) # BxSx1 => BxSxD
        x = x.masked_fill(conv_mask==0, 0)
        x = x.transpose(0, 1) # SxBxH
        x = self.conv(x.contiguous())
        x = x.transpose(0, 1)
        x = self.linear_2(x)
        return x



class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1, weight_softmax=True, weight_dropout=0.):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)

        conv_dim = d_model
        kernel_size = 3
        self.conv = LightweightConvLayer(d_model, conv_dim, kernel_size,
                                            weight_softmax=weight_softmax,
                                            num_heads=heads,
                                            weight_dropout=weight_dropout)

        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # x: BxSxH

        residual = x
        x = self.norm_1(x)
        x = self.conv(x, mask)
        x = self.dropout_1(x)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.ff(x)
        x = self.dropout_2(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1, weight_softmax=True, weight_dropout=0.):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        conv_dim = d_model
        kernel_size = 3
        self.conv = LightweightConvLayer(d_model, conv_dim, kernel_size,
                                            weight_softmax=weight_softmax,
                                            num_heads=heads,
                                            weight_dropout=weight_dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        residual = x
        x = self.norm_1(x)
        x = self.conv(x, trg_mask)
        x = self.dropout_1(x)
        x = residual + x

        residual = x
        x = self.norm_2(x)
        x = self.attn(x,e_outputs,e_outputs,src_mask)
        x = self.dropout_2(x)
        x = residual + x

        residual = x
        x = self.norm_3(x)
        x = self.ff(x)
        x = self.dropout_3(x)
        x = residual + x

        return x


