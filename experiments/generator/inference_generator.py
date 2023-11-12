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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model_name", default="klue/roberta-base",  type=str,  help="which model you train")
parser.add_argument("-f", "--prompt_data", default="a",  type=str, help = "prompt or context")
parser.add_argument("-o", "--output_data", default="a",  type=str, help = "output path")
args = parser.parse_args()
 

model = args.model_name

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(

    "text-generation",
    model=model,
    torch_dtype=torch.float32,
    device_map={"" : 0} # if you have GPU
)
gen_txt = []

prompt_template = """You are an AI assistant for answering questions about technical topics.
    You are given the following extracted parts of long documents and a question. Provide a conversational answer.
    Use the context as a source of information, but be sure to answer the question directly. You're
    job is to provide the user a helpful summary of the information in the context if it applies to the question.
    If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.

    Question:
    Context : 
    """

prompt_data = pd.read_csv(args.prompt_data, sep="\t")
prompt_txt = prompt_data['prompt']
for i in prompt_txt:

    sequences = pipeline(
    prompt_template+i,
    do_sample=True,
    top_k=10,
    top_p = 0.9,
    temperature = 0.2,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200, # can increase the length of sequence
)

    for seq in sequences:
        gen_txt.append(seq['generated_text'])

df = pd.DataFrame()
df['prompt'] = prompt_txt
df['generate'] = gen_txt

df.to_csv(args.output_data,sep='\t')
print(df)