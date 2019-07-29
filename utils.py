import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Transformer.Model.Mask import create_src_mask

def forward(model, src, src_pad_token=0):
    src_mask = create_src_mask(src, pad_token=src_pad_token)
    logit = model(src, src_mask)
    return logit


def forward_and_loss(model, src, trg, loss_fn, src_pad_token=0):     
    src_mask = create_src_mask(src, pad_token=src_pad_token)
    preds = model(src, src_mask)
    ys = trg.contiguous().view(-1)
    loss = loss_fn(preds.view(-1, preds.size(-1)), ys, ignore_index=src_pad_token)
    return preds, loss


def train_model(model, optimizer, train_iter, src_pad_token, save_path=None, device=None):
    total_loss = 0.0
    total_item = 0

    model.train()
    
    with tqdm(total=len(train_iter)) as pbar:
        for src, trg in train_iter: 
            if device is not None and device.type=='cuda':
                src = src.cuda()
                trg = trg.cuda()

            optimizer.zero_grad()
            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad_token=src_pad_token)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_item += trg.size(0)

            pbar.update(1)
            pbar.set_description("loss     = %.8f" % (total_loss/total_item))
            
    # Save model
    if save_path is not None:
        state = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
        }
        
        torch.save(state, save_path)
    
    return total_loss/total_item


def evaluate_model(model, val_iter, src_pad_token, device=None):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0
        total_item = 0
        for src, trg in val_iter:
            if device is not None and device.type=='cuda':
                src = src.cuda()
                trg = trg.cuda()

            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad_token=src_pad_token)
            
            total_loss += loss.item()
            total_item += src.size(0)

            pbar.update(1)
            pbar.set_description("val_loss = %.8f" % (total_loss/total_item))

    return total_loss/total_item

def translate(model, sents, src_tokenizer, trg_tokenizer, maxlen=200, device=None):
    sents = [x.lower().split() for x in sents]
    sents_len =[len(x) for x in sents]
    seqs = []
    for sent in sents:
        seq = [src_tokenizer.word_index[x] if x in src_tokenizer.word_index else 1 for x in sent]
        seqs.append(seq)
    seqs = pad_sequences(seqs, maxlen, padding='post')
    seqs = torch.tensor(seqs).long()
    if device is not None and device.type=='cuda':
        seqs = seqs.cuda()
    with torch.no_grad():
        probs = forward(model, seqs, 0)
    probs = probs.cpu().detach().numpy()
    preds = probs.argmax(axis=-1)
    res = []
    for sent_len, seq in zip(sents_len, preds):
        res.append(seq[:sent_len])
    res = trg_tokenizer.sequences_to_texts(res)
    return res