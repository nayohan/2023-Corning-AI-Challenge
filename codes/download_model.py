import os
from transformers import AutoModel, AutoTokenizer, LlamaTokenizer

save_path = '../pretrained_model'
os.makedirs(save_path, exist_ok=True)


save_t5 = ['t5-small', 't5-base', 't5-large', 't5-3b']#, 't5-11b']
save_flan_t5 = ['google/flan-t5-small', 'google/flan-t5-base', 'google/flan-t5-large', 'google/flan-t5-xl']#, 'google/flan-t5-xxl']
save_llama = ['pinkmanlove/llama-7b-hf', 'pinkmanlove/llama-13b-hf']#, 'pinkmanlove/llama-33b-hf', 'pinkmanlove/llama-65b-hf']
save_llama = ['daryl149/llama-2-7b-hf'] # For tokenizer added eos, bos, pad token

for model_name in save_t5:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{save_path}/{model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(f"{save_path}/{model_name}")

for model_name in save_flan_t5:
    save_model_name = model_name.split('/')[-1]
    save_model_name = save_model_name.replace('xxl', '11b')
    save_model_name = save_model_name.replace('xl', '3b')
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{save_path}/{save_model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(f"{save_path}/{save_model_name}")

    
for model_name in save_llama:
    save_model_name = model_name.split('/')[-1][:-3]
    
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(f"{save_path}/{save_model_name}")
    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(f"{save_path}/{save_model_name}")