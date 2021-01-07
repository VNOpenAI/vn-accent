import argparse
import os
import json
import logging
from pathlib import Path

import torch
from torch.utils import data

from models.model_factory import get_model
from dataloader import Dataset
from utils import train_model, evaluate_model


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('vocab_path')
    parser.add_argument('train_path')
    parser.add_argument('val_path')
    parser.add_argument('--experiment_name')
    parser.add_argument('--src_postfix', default='.notone')
    parser.add_argument('--trg_postfix', default='.tone')
    parser.add_argument('--config_file', default='model_config.json')
    parser.add_argument('--model_name', default='big_evolved')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--restore_file', default=None)

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = get_arg()

    # Init experiment folder
    experiment_folder = os.path.join("experiments", args.experiment_name)
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)

    # Init Log
    log_file = os.path.join(experiment_folder, "logs.txt")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, 
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
    train_dataset = Dataset(src_tokenizer, trg_tokenizer, train_src_file, train_trg_file)
    train_iter = data.dataloader.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_src_file = args.val_path + args.src_postfix
    val_trg_file = args.val_path + args.trg_postfix
    val_dataset = Dataset(src_tokenizer, trg_tokenizer, val_src_file, val_trg_file)
    val_iter = data.dataloader.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model config
    with open(args.config_file) as f:
        config = json.load(f)
    
    if args.model_name in config:
        model_param = config[args.model_name]
    else:
        raise Exception("Invalid model name")
    
    model_param['src_vocab_size'] = len(src_tokenizer.word_index) + 1
    model_param['trg_vocab_size'] = len(trg_tokenizer.word_index) + 1

    # Device 
    print("Init model")
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Init model
    model = get_model(model_param)
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
    weight_folder = os.path.join(experiment_folder, "weights")
    Path(weight_folder).mkdir(parents=True, exist_ok=True)

    # Train model
    print("Start training %d epochs" % args.num_epochs)
    for e in range(1, args.num_epochs+1):
        logger.info("Epoch %02d/%02d" % (e, args.num_epochs))
        logger.info("Start training")
        print("\nEpoch %02d/%02d" % (e, args.num_epochs), flush=True)
        save_file = os.path.join(weight_folder, 'epoch_%02d.h5' % e)
        train_loss = train_model(model, optim, train_iter, src_pad_token, use_mask=model_param["use_mask"], device=device, save_path=save_file)
        logger.info("End training")
        logger.info("train_loss = %.8f" % train_loss)
        val_loss = evaluate_model(model, val_iter, src_pad_token, use_mask=model_param["use_mask"], device=device)
        logger.info("val_loss   = %.8f\n" % val_loss)