import sys
import argparse
import gc

import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import numpy as np

from transformers import AutoConfig, AutoTokenizer, AutoModel
from contrastive_learning.utils import *
from sklearn import metrics


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)    
    p.add_argument('--test_fn', required=True)
    p.add_argument('--pooler_type', type=str, choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], default='cls')
    
    p.add_argument('--temp', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--top_k', type=int, default=5)
    
    args = p.parse_args()

    return args

def main(args):
    checkpoint = torch.load(args.model_fn, map_location='cpu' if args.gpu_id < 0 else 'cuda:%d' % args.gpu_id)
    config = checkpoint['args']
    model_best = checkpoint['model_best']
    
    dataset = pd.read_csv(args.test_fn, sep='\t', encoding='utf-8')

    with torch.no_grad():        
        tokenizer = AutoTokenizer.from_pretrained(config.model_path_or_name)
        
        model = AutoModel.from_pretrained(config.model_path_or_name)
        model.load_state_dict(model_best)
    
        if args.gpu_id >= 0:
            model.cuda(args.gpu_id)
        
        device = next(model.parameters()).device
        
        queries, references = dataset['question'].values, dataset['references'].values
        queries = [str(x) for x in queries]
        documents = [x for x in references]
        
        model.eval()
        encoding_q = tokenizer(queries, 
                            padding=True, 
                            truncation=True, 
                            return_tensors='pt', 
                            max_length=args.max_length)
        
        encoding_d = tokenizer(documents, 
                            padding=True, 
                            truncation=True, 
                            return_tensors='pt', 
                            max_length=args.max_length)
    
        x_q, mask_q = encoding_q['input_ids'].to(device), encoding_q['attention_mask'].to(device)
        x_d, mask_d = encoding_d['input_ids'].to(device), encoding_d['attention_mask'].to(device)

        output_q = model(x_q, attention_mask=mask_q)
        output_d = model(x_d, attention_mask=mask_d)
        
        pooler = Pooler(args.pooler_type)
        z_q = pooler(mask_q, output_q)
        z_d = pooler(mask_d, output_d)
        
        sim = Similarity(temp=1)
        cos_sim = sim(z_q.unsqueeze(1), z_d.unsqueeze(0))
        
        true = torch.diag(torch.eye(cos_sim.size(0))).cpu().numpy()
        pred = topk_metric(cos_sim.cpu().detach().numpy(), k=args.top_k)
        
        recall = metrics.recall_score(true, pred) * 100
        f1_score = metrics.f1_score(true, pred) * 100
        print('\n----------------------')
        print('|| recall@{} | {:.1f}% ||'.format(args.top_k, recall))
        print('----------------------')
        print('|| f1_score | {:.1f}% ||'.format(f1_score))
        print('----------------------\n')
        
if __name__ == '__main__':
    args = define_argparser()
    main(args)
    