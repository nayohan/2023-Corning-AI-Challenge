import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


load_data_path = './'
save_data_path = './tasks/DocQA7_MS'

df = pd.read_json('con_quer_En3.jsonl', orient='records', lines=True)


guideline = "You will be shown a dialogues between Speaker 1 and Speaker 2. Please read Context and understand given Dialogue Session, then complete the task under the guidance of Task Introduction.\n\n"
task_introduction = "```\nTask Introduction:\nAfter reading the Dialogue Session, please create an appropriate response in the parts marked ###.\n```\n\nTask Result:"

task_dsg = []
for idx in tqdm(range(len(df))): # number of data
	multi_turn_dialogue = []
	n_turn = len(df['dialogue'][idx])

	for turn in range(n_turn):
		row = f"{df['speaker'][idx][turn]}: {df['dialogue'][idx][turn]}\n"
		multi_turn_dialogue.append(row)

	last_response = multi_turn_dialogue[-1].split(':')[-1]
	last_spaker = df['speaker'][idx][-1]
	multi_turn_dialogue[-1] = last_spaker + ': ###\n'
	dialogue_session = ''.join(multi_turn_dialogue)

	context  = "```\nContext:\n" + df['context'][idx] + "```\n\n"
	dialogue = "```\nDialogue Session:\n" + dialogue_session + "```\n\n"

	input = guideline + context + dialogue + task_introduction
	output = last_response
	task_dsg.append([input, output])

preprocessed_df = pd.DataFrame(task_dsg, columns=['input', 'output'])
print(df)
print(preprocessed_df)

train, valid = train_test_split(preprocessed_df, test_size=0.05, random_state=2024)
os.makedirs(save_data_path, exist_ok=True)
train.to_json(f'{save_data_path}/train.jsonl', orient='records', lines=True)
valid.to_json(f'{save_data_path}/test.jsonl', orient='records', lines=True)