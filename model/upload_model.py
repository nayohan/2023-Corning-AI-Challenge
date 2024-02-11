from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def save_model_tokenizer(model_path, save_name):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.push_to_hub(save_name)
    tokenizer.push_to_hub(save_name)
 
if __name__=="__main__":
    model_path = './meta-llama/Llama-2-13b-chat-hf_DocQA7_MS'
    save_name = 'corningQA-llama2-13b-chat'
    save_model_tokenizer(model_path, save_name)

    model_path = './upstage/SOLAR-10.7B-v1.0_DocQA7_MS'
    save_name = 'corningQA-solar-10.7b-v1.0'
    save_model_tokenizer(model_path, save_name)