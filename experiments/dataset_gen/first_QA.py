import pandas as pd
import openai
import time
from tqdm import tqdm
# from openai import OpenAI
# client = OpenAI()

openai.api_key = "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"
con = pd.read_json("/home/dilab/jam/rag2/ClosedAI/experiments/dataset_gen/context.jsonl", orient="records", lines=True)

query = []
context = []
answer = []
for i in tqdm(range(len(con))):
    messages = []
    prompt = """
Context:
{}

질문 : ###
답변: ###

Task Introduction:
위의 Context를 보고 해당 내용에 대한 질문과 답변을 한글로 생성해줘
"""

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt.format(con['context'][i]),
    max_tokens = 200
    )

    
    chat_response = response.choices[0].text
    print(chat_response)

    try:

        output_txt=chat_response.replace("\n질문: ","").replace("\n질문 : ","").replace("\n답변: ",">>").replace("\n답변 : ",">>").split(">>")
    # 
        answer.append(output_txt[1])
        query.append(output_txt[0])
        context.append(con['context'][i])

    except IndexError:
        pass

    output = pd.DataFrame()
    output['context']=context
    output['query'] = query
    output['answer'] = answer
    output.to_json("con_quer.jsonl", orient="records", lines=True)


    


