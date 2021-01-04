import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def nopeak_mask(size):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = np_mask == 0
    np_mask =  torch.from_numpy(np_mask.astype('uint8'))
    return np_mask


def create_src_mask(src, pad_token):
    """ src: BxS
        pad_token: index of pad_token token

        output: Bx1xS --> broadcast BxSxS
    """
    src_mask = (src != pad_token).unsqueeze(-2)
    return src_mask

def create_trg_mask(trg, pad_token):
    trg_mask = (trg != pad_token).unsqueeze(-2)
    size = trg.size(1) # get seq_len for matrix
    np_mask = nopeak_mask(size)
    if trg.is_cuda:
        np_mask = np_mask.cuda()
    trg_mask = trg_mask & np_mask
    return trg_mask

def create_mask(src, trg, src_pad_token, trg_pad_token):
    src_mask = create_src_mask(src, src_pad_token)
    trg_mask = create_trg_mask(trg, trg_pad_token)
    return src_mask, trg_mask