import argparse
import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

from model import get_model
from dataloader import MTDataset
from utils import train_model, evaluate_model


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_path')
    parser.add_argument('train_path')
    parser.add_argument('val_path')
    parser.add_argument('--src_postfix', default='.notone')
    parser.add_argument('--trg_postfix', default='.tone')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--weight_dir', default='weight')
    parser.add_argument('--restore_file', default=None)
    parser.add_argument('--log_file', default='log.txt')

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = get_arg()

    # Init Log
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, 
                        filemode='a',
                        level=logging.INFO, 
                        format="%(levelname)s - %(asctime)s: %(message)s")
    logger=logging.getLogger(__name__)

    # Load tokenizer
    print("Load tokenizer")
    tokenizer = torch.load(args.vocab_path)
    src_tokenizer = tokenizer['notone']
    trg_tokenizer = tokenizer['tone']
    src_pad_token = 0
    trg_pad_token = 0

    # Load data
    print("Load data")
    train_src_file = args.train_path + args.src_postfix
    train_trg_file = args.train_path + args.trg_postfix
    train_dataset = MTDataset(src_tokenizer, trg_tokenizer, train_src_file, train_trg_file)
    train_iter = data.dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_src_file = args.val_path + args.src_postfix
    val_trg_file = args.val_path + args.trg_postfix
    val_dataset = MTDataset(src_tokenizer, trg_tokenizer, val_src_file, val_trg_file)
    val_iter = data.dataloader.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model config
    evolved_big_param = {
        "src_vocab_size": len(src_tokenizer.word_index) + 1,
        "trg_vocab_size": len(trg_tokenizer.word_index) + 1,
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 3,
        "num_heads": 16,
        "dropout": 0.3,
        "is_evolved": True,
        "only_encoder": True
    }

    # Device 
    print("Init model")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Init model
    model = get_model(**evolved_big_param)
    if device.type=='cuda':
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    print("Using", device.type)

    # Load weight
    if args.restore_file is not None:
        if os.path.isfile(args.restore_file):
            print("Load model")
            state = torch.load(args.restore_file)
            model.load_state_dict(state['model'])
            optim.load_state_dict(state['optim'])
        else:
            raise Exception("Invalid weight path")
    
    # Init weight dir
    if not os.path.isdir(args.weight_dir):
        os.makedirs(args.weight_dir)

    # Train model
    print("Start training %d epochs" % args.num_epochs)
    for e in range(1, args.num_epochs+1):
        logger.info("Epoch %02d/%02d" % (e, args.num_epochs))
        logger.info("Start training")
        print("\nEpoch %02d/%02d" % (e, args.num_epochs), flush=True)
        save_file = os.path.join(args.weight_dir, 'epoch_%02d.h5' % e)
        train_loss = train_model(model, optim, train_iter, src_pad_token, device=device, save_path=save_file)
        logger.info("End training")
        logger.info("train_loss = %.8f" % train_loss)
        val_loss = evaluate_model(model, val_iter, src_pad_token, device=device)
        logger.info("val_loss   = %.8f\n" % val_loss)