import os
import argparse
import random
import jsonlines
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers.trainer_utils import set_seed
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from ignite.utils import manual_seed

from contrastive_learning.utils import *
from contrastive_learning.trainer import Trainer
from contrastive_learning.dataset import *

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--output_path', type=str, required=True)

    p.add_argument("--train_fn", type=str, required=True)
    p.add_argument("--valid_fn", type=str, required=True)
    
    p.add_argument("--model_path_or_name", type=str, default='bert-base-uncased')
    
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--eval_step', type=int, default=50)

    p.add_argument('--n_epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--pooler_type', type=str, 
                                    choices=["cls", "cls_before_pooler", "avg", "avg_top2", "avg_first_last"], 
                                    default='cls')
    
    p.add_argument('--temp', type=float, default=0.06949)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--warmup_ratio', type=float, default=0)
    
    p.add_argument('--mlm_weight', type=float, default=0.1)

    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--do_mlm', action='store_true')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--max_length', type=int, default=64)    
    
    args = p.parse_args()

    return args
    
def init_all(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    manual_seed(seed)
    random.seed(seed)
    torch.cuda.empty_cache()

def _jsonlines(file, mode='train'):
    sent1, sent2, score = [], [], []
    with jsonlines.open(file) as f:
        if mode == 'train':
            for line in f:
                sent1.append(line['references'])
                sent2.append(line['question'])
            return sent1, sent2
            
        elif mode == 'valid':
            for line in f:
                sent1.append(line['sentence1'])
                sent2.append(line['sentence2'])
                score.append(line['score'])
            return sent1, sent2, score
            
def CL_get_loaders(args, tokenizer):
    sent1, sent2 = _jsonlines(args.train_fn, mode='train')
    sent1_, sent2_, score = _jsonlines(args.valid_fn, mode='valid')
    
    train_dataset = list(zip(sent1, sent2))
    valid_dataset = list(zip(sent1_, sent2_, score))
    
    sent1 = [e[0] for e in train_dataset]
    sent2 = [e[1] for e in train_dataset]
    
    sent1_ = [e[0] for e in valid_dataset]
    sent2_ = [e[1] for e in valid_dataset]    
    score = [e[2] for e in valid_dataset]
    
    train_loader = DataLoader(
        TrainDataset(sent1, sent2),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer, args, mode="train"),
    )

    valid_loader = DataLoader(
        ValidDataset(sent1_, sent2_, score),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DataCollator(tokenizer, args, mode="valid"),
    )
    return train_loader, valid_loader    


def get_optimizer(model, args):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        eps=args.adam_epsilon
    )

    return optimizer

def main(args):
    init_all(args.seed)
    set_seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
    
    model = AutoModel.from_pretrained(args.model_path_or_name).cuda(args.gpu_id)
    
    config = AutoConfig.from_pretrained(args.model_path_or_name)
    train_loader, valid_loader = CL_get_loaders(args, tokenizer)    
    
    print(
        '|train| =', len(train_loader) * args.batch_size,
        '|valid| =', len(valid_loader) * args.batch_size,
    )
    n_total_iterations = len(train_loader) * args.n_epochs
    n_warmup_steps = int(n_total_iterations * args.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    optimizer = get_optimizer(model, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        n_warmup_steps,
        n_total_iterations
    )
    if args.gpu_id >= 0:
        model.cuda(args.gpu_id)

    trainer = Trainer(args, config)
    model = trainer.train(
                model,
                optimizer,
                scheduler,
                train_loader,
                valid_loader,
                )

if __name__ == '__main__':
    args = define_argparser()
    main(args)
    