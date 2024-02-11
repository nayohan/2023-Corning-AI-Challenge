import os
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer,LlamaForCausalLM, LlamaConfig
from datasets import load_dataset, Dataset, DatasetDict
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

data_path = "/home/user/ClosedAI/app/test.jsonl"

# model_name = "/home/uj-user/Yo/hybrid-ltm/model/llama-7b_ALL"
# model_name = "/home/uj-user/Yo/hybrid-ltm/model/llama-7b_MS_DG"
# model_name = "gpt2"
# model_name = "nayohan/closedai-llm"
# model_name = "nayohan/corningQA-llama2-13b-chat"
# model_name = "nayohan/corningQA-solar-10.7b-v1.0"
model_name ="lmsys/vicuna-7b-v1.5"
max_len = 2048
target_len = 128
device = "cuda:1" if torch.cuda.is_available() else "cpu"

config = LlamaConfig.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, config=config).to(device)#, torch_dtype=torch.bfloat16).to(device)
dataset = pd.read_json(data_path, lines=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, truncation_side='left', use_fast=False)
# tokenizer.add_special_tokens({'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'})

ans_jsons = []
for i, line in enumerate(tqdm(dataset['input'])):
    # print(dataset['input'][i])
    input = tokenizer([dataset['input'][i]], max_length=(max_len-target_len), truncation=True, return_tensors='pt').to(device)#.input_ids
    target_len = min(len(input.input_ids[0])+target_len, max_len)
    output_ids = model.generate(
        **input,
        do_sample=True,
        max_length=768,
        encoder_no_repeat_ngram_size=None
    )
    result_ids = output_ids[0][len(input.input_ids[0]):]
    outputs = tokenizer.decode(result_ids, skip_special_tokens=False).strip()
    # print('input:\n', tokenizer.decode(input.input_ids[0]))
    # print('output:\n', outputs)
    if i==10:
        break
    print(i)
    ans_jsons.append(
        {
            "question_id": i,
            "text": outputs,
            "model_id": model_name,
            "metadata": {},
        }
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True).strip())
print(outputs)

filename = f"predictions/{model_name}_{data_path.split('/')[-2]}.json"
save_path = os.path.dirname(filename)
os.makedirs(save_path, exist_ok=True)
df_ans_json = pd.DataFrame(ans_jsons)
df_ans_json.to_json(filename, orient='records', lines=True)