from transformers import AutoTokenizer
import transformers
import torch
from dataclasses import dataclass, field
import logging
import math
import os
import sys
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import json
from rouge import Rouge
from bert_score import score


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", default="klue/roberta-base",  type=str,  help="which model you train")
parser.add_argument("--evaluation_file", required=True)
parser.add_argument("-f", "--query_and_ref", default="a",  type=str, help = "query and references")
# parser.add_argument("-o", "--output_data", default="a",  type=str, help = "output path")
# parser.add_argument("-p", "--prompt_template", default="a",  type=str, help = "prompt template")

args = parser.parse_args()

dataset = load_dataset(args.evaluation_file)
checkpoint = args.model_name
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if 't5' in checkpoint:
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
else:
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

data = dataset['test'][:3]

gen_txt =[] 
gen_txtp =[] 
for i in range(len(data)):
    prompt = """
    Question: {}
    =========
    {}
    =========
    Answer in:
    """



    if 't5' in checkpoint:
        prompt = prompt.format(data['question'][i], data['references'][i])
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
        generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(generation)
        gen_txtp.append(generation[0])
    
    else:
        prompt = prompt.format(data['question'][i], data['references'][i])
        inputs = tokenizer(prompt, return_tensors="pt")
        length = len(prompt)
        max_l = length + 128
        outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id)
        generation = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(generation)
        gen_txtp.append(generation[0][len(prompt):])


rouge = Rouge()

print("rouge score : " , rouge.get_scores(gen_txtp, data['answer'], avg=True))
P, R, F1 =score(gen_txtp, data['answer'], lang='en', verbose=True)
print("Bert_score : ", f"System level F1 score: {F1.mean():.3f}")