import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences
from accent_utils import extract_words, remove_tone_line
from models.Mask import create_src_mask

def forward(model, src, src_pad_token=0, use_mask=True):
    if use_mask:
        src_mask = create_src_mask(src, pad_token=src_pad_token)
        logit = model(src, src_mask)
    else:
        logit = model(src)
    return logit


def forward_and_loss(model, src, trg, loss_fn, src_pad_token=0, use_mask=True):
    if use_mask:
        src_mask = create_src_mask(src, pad_token=src_pad_token)
        preds = model(src, src_mask)
    else:
        preds = model(src)
    ys = trg.contiguous().view(-1)
    loss = loss_fn(preds.view(-1, preds.size(-1)), ys, ignore_index=src_pad_token)
    return preds, loss


def train_model(model, optimizer, train_iter, src_pad_token, use_mask=True, save_path=None, device=None):
    total_loss = 0.0
    total_item = 0

    model.train()
    
    with tqdm(total=len(train_iter)) as pbar:
        for src, trg in train_iter: 
            if device is not None and device.type=='cuda':
                src = src.cuda()
                trg = trg.cuda()

            optimizer.zero_grad()
            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, src_pad_token=src_pad_token, use_mask=use_mask)
            
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


def evaluate_model(model, val_iter, src_pad_token, use_mask=True, device=None):
    model.eval()
    with torch.no_grad(), tqdm(total=len(val_iter)) as pbar:
        total_loss = 0.0
        total_item = 0
        for src, trg in val_iter:
            if device is not None and device.type=='cuda':
                src = src.cuda()
                trg = trg.cuda()

            _, loss = forward_and_loss(model, src, trg, F.cross_entropy, use_mask=use_mask, src_pad_token=src_pad_token)
            
            total_loss += loss.item()
            total_item += src.size(0)

            pbar.update(1)
            pbar.set_description("val_loss = %.8f" % (total_loss/total_item))

    return total_loss/total_item

def translate(model, sents, src_tokenizer, trg_tokenizer, maxlen=200, use_mask=True, device=None):
    
    words, word_indices = extract_words(sents)
    lower_words = [x.lower() for x in words]

    # Tokenize words
    known_word_mask = [] # Same size as words - True if word is in word list, otherwise False
    seqs = []
    for word in lower_words:
        if word in src_tokenizer.word_index:
            seqs.append(src_tokenizer.word_index[word])
            known_word_mask.append(True)
        else:
            seqs.append(1)
            known_word_mask.append(False)
    seqs = [seqs]

    # Model inference
    seqs = pad_sequences(seqs, maxlen, padding='post')
    seqs = torch.tensor(seqs).long()
    if device is not None and device.type=='cuda':
        seqs = seqs.cuda()
    with torch.no_grad():
        probs = forward(model, seqs, 0, use_mask=use_mask)
    probs = probs.cpu().detach().numpy()
    
    # Add tone
    output = sents
    probs = probs[0]
    prob_indices = probs.argsort(axis=-1)[:, ::-1]
    prob_indices = prob_indices[:, :100]
    for i, word in enumerate(lower_words):
        
        # Skip unknown words
        if not known_word_mask[i]:
            continue

        # Find the best solution
        for idx in prob_indices[i, :]:
            target_word = trg_tokenizer.sequences_to_texts([[idx]])[0]
            if remove_tone_line(target_word.lower()) == word:
                begin_idx, end_idx = word_indices[i]

                # Correct lower / upper case
                corrected_word = ""
                for ic, char in enumerate(words[i]):
                    if char.islower():
                        corrected_word += target_word[ic].lower()
                    else:
                        corrected_word += target_word[ic].upper()

                output = output[:begin_idx] + corrected_word + output[end_idx:]
                break

    return output