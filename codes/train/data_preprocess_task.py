import copy

from typing import Optional
from dataclasses import dataclass, field

import torch
from datasets import load_dataset
from transformers import HfArgumentParser, T5Tokenizer, LlamaTokenizer, set_seed

q_pre = "<s>\n"
qa_link = "\n"
a_pos = "\n</s>"

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainingArguments:
    data_path: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    preprocessed_path: str = field(
        default=None, metadata={"help": "Path to the preprocessed training data."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    
    raw_datasets = load_dataset(data_args.data_path)
    column_names = raw_datasets["train"].column_names
    print("load dataset finished")

    if "t5" in model_args.model_name_or_path:
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = T5Tokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left')

        def preprocess_function(examples):
            src_inputs = [q_pre + example + qa_link for example in examples["input"]]
            src_model_inputs = tokenizer(src_inputs, max_length=data_args.model_max_length, padding='longest', truncation=True, add_special_tokens=False)
            trg_inputs = [example + a_pos for example in examples["output"]]
            trg_model_inputs = tokenizer(trg_inputs, max_length=data_args.model_max_length, padding='longest', truncation=True, add_special_tokens=False)
            src_model_inputs["labels"] = [
                [(l if l != tokenizer.pad_token_id else label_ignore_id) for l in label] for label in trg_model_inputs["input_ids"]
            ]
            return src_model_inputs
    else:
        # use truncation_side='left' to preserve linking between end of prompt and target labels
        tokenizer = LlamaTokenizer.from_pretrained(model_args.model_name_or_path, truncation_side='left')
        tokenizer.add_special_tokens({'pad_token': '</s>'})

        def preprocess_function(examples):
            inputs = [q_pre + input + qa_link + output + a_pos for input, output in zip(examples["input"], examples["output"])]
            model_inputs = tokenizer(inputs, max_length=data_args.model_max_length, padding="longest", truncation=True, add_special_tokens=False)
            model_inputs["labels"] = copy.deepcopy(model_inputs["input_ids"])
            for e_i, (input, output) in enumerate(zip(examples["input"], examples["output"])):
                # print('input:',input.shape)
                # print('output:',output.shape)
                source_text = q_pre + input + qa_link
                target_text = output + a_pos
                source_ids = tokenizer.encode(source_text, add_special_tokens=False)
                target_ids = tokenizer.encode(target_text, add_special_tokens=False)                
                # print('input:',len(source_ids))
                # print('output:',len(target_ids))
                if len(source_ids) >= data_args.model_max_length:
                    model_inputs["labels"][e_i] = [label_ignore_id] * data_args.model_max_length
                    continue
                else:
                    model_inputs["labels"][e_i][:len(source_ids)] = [label_ignore_id] * len(source_ids)
                    if len(target_ids) + len(source_ids) >= len(model_inputs["input_ids"][e_i]):
                        continue
                    else:
                        model_inputs["labels"][e_i][(len(target_ids) + len(source_ids)):] = [label_ignore_id] * (len(model_inputs["input_ids"][e_i]) - len(target_ids) - len(source_ids))
            model_inputs["input_ids"] = torch.tensor(model_inputs["input_ids"])
            model_inputs["labels"] = torch.tensor(model_inputs["labels"])
            print('model_inputs["input_ids"]:',model_inputs["input_ids"].shape)
            print('model_inputs["labels"]:',model_inputs["labels"].shape)

            model_inputs["attention_mask"] = model_inputs["input_ids"].ne(tokenizer.pad_token_id)
            return model_inputs
    
    label_ignore_id = -100

    print("start data preprocess")
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=len(train_dataset),
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset"
    )
    train_dataset.save_to_disk(data_args.preprocessed_path+'/train')
    
    validation_dataset = raw_datasets["validation"]
    validation_dataset = validation_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=len(validation_dataset),
        remove_columns=column_names,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc="Running tokenizer on train dataset"
    )
    validation_dataset.save_to_disk(data_args.preprocessed_path+'/validation')
    
    print(data_args.preprocessed_path+'/validation')
    print("data preprocess finished")

if __name__ == "__main__":
    main()