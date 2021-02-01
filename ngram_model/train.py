import argparse
from torch import load
from tqdm import tqdm
import pickle
from nltk.tokenize import word_tokenize
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import KneserNeyInterpolated
import time


def get_arg():
    parse = argparse.ArgumentParser()
    parse.add_argument('doc_dir')
    parse.add_argument('model_dir')
    parse.add_argument('--ngram', type=int, default=3)
    return parse.parse_args()

# tokenize word, if word not in dictionary then word = 'unknown'
def tokenize(doc):
    temp = load('tokenizer_standarized.h5')
    vnword = temp['tone'].word_index
    result = []
    for sent in tqdm(doc):
        temp = word_tokenize(sent)
        for idx, word in enumerate(temp):
            if word not in vnword:
                temp[idx] = 'unknown'
        result.append(temp)
    print('tokenize done')
    return result
    

if __name__=='__main__':
    arg = get_arg()

    # get train data and tokenize
    with open(arg.doc_dir, 'r', encoding='utf-8') as fin:
        doc = fin.readlines()
    corpus = tokenize(doc)
    del doc

    vi_model = KneserNeyInterpolated(arg.ngram)
    train_data, padded_sent = padded_everygram_pipeline(arg.ngram, corpus)
    del corpus
    start_time = time.time()
    vi_model.fit(train_data, padded_sent)
    print('train %s-gram model in %d s'%(arg.ngram, time.time()-start_time))
    print('length of vocab = %s'%(len(vi_model.vocab)))

    with open(arg.model_dir, 'wb') as fout:
        pickle.dump(vi_model, fout)
    print('save model successfully!')
  


  