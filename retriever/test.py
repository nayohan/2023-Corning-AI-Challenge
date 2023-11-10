import sys
import argparse
import gc

import torch 
import torch.nn as nn
import numpy as np

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModel
from simcse import SimCSE

from utils import Pooler, Similarity
    
def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_name_or_path', required=True)
    p.add_argument('--simcse_model_name_or_path', required=True)
    
    p.add_argument('--evaluation_file', required=True)
    p.add_argument('--data_type', type=str, choices=['train', 'validation', 'test'], default='test')
    p.add_argument('--pooler_type', type=str, choices=['cls', 'cls_before_pooler', 'avg', 'avg_top2', 'avg_first_last'], default='cls')
    
    p.add_argument('--do_simcse', action='store_true', default=False)
    p.add_argument('--temp', type=int, default=1)
    p.add_argument('--max_length', type=int, default=64)
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--use_fast_tokenizer', action='store_true')

    args = p.parse_args()

    return args

def topk_metric(matrix, k=None):
    result = []
    for c in matrix:
        topk = sorted(c, reverse=True)[:k]
        diag = np.diag(matrix)
        is_topk = any(x in topk for x in np.diag(matrix))
        result.append(0 if is_topk else 1) # 정답이면 0 아니면 1
    
    return result

def main(args):
    dataset = load_dataset(args.evaluation_file)[args.data_type]
    
    with torch.no_grad():
        
        tokenizer_kwargs = {"use_fast" : args.use_fast_tokenizer}
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
        
        if args.do_simcse:
            model = SimCSE(args.simcse_model_name_or_path)
        else:
            if args.gpu_id >= 0:
                model.cuda(args.gpu_id)
            
            device = next(model.parameters()).device

            model_loader = AutoModel
            model = model_loader.from_pretrained(args.model_name_or_path)
            
        queries, references = dataset['question'], dataset['references']
        documents = [' '.join(x) for x in references]
        del dataset
        del references
        gc.collect()
        
        if args.do_simcse:
            output_q = model.encode(queries)
            output_d = model.encode(documents)
        
        else:
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

            output_q = model(x_q, mask_q)
            output_d = model(x_d, mask_d)
            
            pooler = Pooler(args.pooler_type)
            z_q = pooler(mask_q, output_q)
            z_d = pooler(mask_d, output_d)
            
        sim = Similarity(args.temp)
        
        if args.do_simcse:
            cos_sim = sim(output_q.unsqueeze(1), output_d.unsqueeze(0))
        
        else:
            cos_sim = sim(z_q.unsqueeze(1), z_d.unsqueeze(0))
        
        result_lst = topk_metric(cos_sim.cpu().detach().numpy(), k=args.top_k)
        acc = result_lst.count(0) / len(result_lst) * 100
        print(acc)

if __name__ == '__main__':
    args = define_argparser()
    main(args)