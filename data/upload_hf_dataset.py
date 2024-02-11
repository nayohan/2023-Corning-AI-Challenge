
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

train_df = pd.read_json('/home/user/ClosedAI/data/tasks/DocQA7_MS/train.jsonl', lines=True, orient='records')
valid_df = pd.read_json('/home/user/ClosedAI/data/tasks/DocQA7_MS/valid.jsonl', lines=True, orient='records')
test_df = pd.read_json('/home/user/ClosedAI/data/tasks/DocQA7_MS/test.jsonl', lines=True, orient='records')

train = Dataset.from_pandas(train_df)
valid = Dataset.from_pandas(valid_df)
test = Dataset.from_pandas(test_df)

DocQA7 = DatasetDict({'train':train, 'validation':valid, 'test':test})
DocQA7.push_to_hub('myngsoooo/CorningAI-DocQA')