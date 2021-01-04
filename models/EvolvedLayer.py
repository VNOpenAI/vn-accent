import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .UtilLayer import *

class SeparableConv1D(nn.Module):
    """ Input: (batch_size, in_channel, length)
        Output: (batch_size, out_channel, length)
    """
    def __init__(self, in_channel, out_channel, kernel_size=1, padding=0):
        super().__init__()
        self.deep_wise = nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, padding=padding, groups=in_channel)
        self.point_wise = nn.Conv1d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        x = self.deep_wise(x)
        x = self.point_wise(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # GLU
        self.norm_glu = Norm(d_model)
        self.glu_ff1 = nn.Linear(d_model, d_model)
        self.glu_ff2 = nn.Linear(d_model, d_model)
        
        # Conv
        self.norm_conv1 = Norm(d_model)
        self.norm_conv2 = Norm(d_model*4)

        self.left_conv = nn.Linear(d_model, d_model * 4)
        self.left_dropout = nn.Dropout(dropout)

        self.right_conv = nn.Conv1d(d_model, d_model//2, kernel_size=3, padding=1)
        self.right_dropout = nn.Dropout(dropout)

        self.sep_conv = SeparableConv1D(d_model*4, d_model//2, kernel_size=9, padding=4)

        # Self-attention
        self.norm_attn = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)

        # Fully connected
        self.norm_ff = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)


    def forward(self, x, mask):
        # GLU: 512
        residual = x
        x = self.norm_glu(x)
        values = self.glu_ff1(x)
        gates = torch.sigmoid(self.glu_ff2(x))
        hiddent_state = values * gates
        x = residual + hiddent_state

        # Conv: 512
        conv_mask = mask[:,-1,:] # BxS
        conv_mask = conv_mask.unsqueeze(-1) # BxSx1 => BxSxD

        residual = x
        x = self.norm_conv1(x)
        x = x.masked_fill(conv_mask==0, 0)

        left_state = self.left_conv(x)
        left_state = F.relu(left_state)
        left_state = self.left_dropout(left_state) # 2048

        right_state = self.right_conv(x.transpose(-1,-2)).transpose(-1,-2)
        right_state = F.relu(right_state)
        right_state = self.right_dropout(right_state) # 256

        right_state = F.pad(right_state, (0, self.d_model*4 - self.d_model//2))
        hiddent_state = left_state + right_state # 2048

        hiddent_state = self.norm_conv2(hiddent_state) 
        hiddent_state = hiddent_state.masked_fill(conv_mask==0, 0)

        hiddent_state = self.sep_conv(hiddent_state.transpose(-1,-2)).transpose(-1,-2) # 256
        hiddent_state = F.pad(hiddent_state, (0, self.d_model//2)) # 512

        x = residual + hiddent_state # 512

        # Self-attention: 512
        residual = x
        x = self.norm_attn(x)
        attn = self.attn(x, x, x, mask)
        attn = self.attn_dropout(attn)
        x = residual + attn

        # Fully connected: 512
        residual = x
        x = self.norm_ff(x)
        hiddent_state = self.ff(x)
        x = residual + hiddent_state

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

        self.heads = heads
        self.d_model = d_model

        # Attention 1
        self.norm_attn_1 = Norm(d_model)
        self.self_attn_1 = MultiHeadAttention(heads*2, d_model, dropout=dropout)
        self.self_attn_dropout_1 = nn.Dropout(dropout)
        self.enc_attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.enc_attn_dropout_1 = nn.Dropout(dropout)

        # Conv
        self.norm_conv1 = Norm(d_model)
        self.norm_conv2 = Norm(d_model*2)
        
        self.left_sep_conv = SeparableConv1D(d_model, d_model*2, kernel_size=11)
        self.right_sep_conv = SeparableConv1D(d_model, d_model//2, kernel_size=7)
        self.sep_conv = SeparableConv1D(d_model*2, d_model, kernel_size=7)

        # Attention 2
        self.norm_attn_2 = Norm(d_model)
        self.self_attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        # Attention 3
        self.norm_attn_3 = Norm(d_model)
        self.enc_attn_3 = MultiHeadAttention(heads, d_model, dropout=dropout)

        # Feed Forward
        self.norm_ff = Norm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout, activation=lambda x: x*torch.sigmoid(x))

    def forward(self, x, e_outputs, src_mask, trg_mask):
        # Attention 1: 512
        residual = x
        x = self.norm_attn_1(x)
        
        self_attn = self.self_attn_1(x, x, x, trg_mask)
        self_attn = self.self_attn_dropout_1(self_attn)

        enc_attn = self.enc_attn_1(x, e_outputs, e_outputs, src_mask)
        enc_attn = self.enc_attn_dropout_1(enc_attn)

        hiddent_state = self_attn + enc_attn
        x = residual + hiddent_state

        # Conv: 512
        conv_mask = trg_mask[:,-1,:] # BxS
        conv_mask = conv_mask.unsqueeze(-1) # BxSx1 => BxSxD

        residual = x
        x = self.norm_conv1(x)
        x = x.masked_fill(conv_mask==0, 0)

        x_pad = F.pad(x.transpose(-1,-2), (10, 0))
        left_state = self.left_sep_conv(x_pad).transpose(-1,-2) # 1024
        left_state = F.relu(left_state)
        
        x_pad = F.pad(x.transpose(-1,-2), (6, 0))
        right_state = self.right_sep_conv(x_pad).transpose(-1,-2) # 256

        right_state = F.pad(right_state, (0, self.d_model*2 - self.d_model//2)) # 1024
        hiddent_state = left_state + right_state # 1024

        hiddent_state = self.norm_conv2(hiddent_state) # 512
        hiddent_state = hiddent_state.masked_fill(conv_mask==0, 0)
        hiddent_state_pad = F.pad(hiddent_state.transpose(-1,-2), (6, 0))
        hiddent_state = self.sep_conv(hiddent_state_pad).transpose(-1,-2)

        x = residual + hiddent_state # 512
        
        # Attention 2
        residual = x
        x = self.norm_attn_2(x)
        # x = x.masked_fill(conv_mask==0, 0)

        self_attn = self.self_attn_2(x, x, x, trg_mask)

        x = residual + self_attn

        # Attention 3
        residual = x
        x = self.norm_attn_3(x)

        enc_attn = self.enc_attn_3(x, e_outputs, e_outputs, src_mask)

        x = residual + enc_attn

        # Feed Forward
        residual = x
        x = self.norm_ff(x)
        hiddent_state = self.ff(x)
        x = residual + hiddent_state

        return x