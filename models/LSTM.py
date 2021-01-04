import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UtilLayer import *


class CNN(nn.Module):
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
