import os
from typing import List, TYPE_CHECKING
import torch
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI,ChatOllama
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.llms import HuggingFaceHub
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

# from qna.constants import (
#     OPENAI_COMPLETIONS_ENGINE,
#     OPENAI_EMBEDDINGS_ENGINE,
# )

if TYPE_CHECKING:
    from langchain.vectorstores import Chroma

from langchain.llms import HuggingFaceHub, HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain import LLMChain
import transformers

def get_llm(max_tokens=100) -> LLM:
    # llm = ChatOpenAI(model_name=OPENAI_COMPLETIONS_ENGINE, max_tokens=max_tokens)
    # llm = HuggingFaceHub(model_name="meta-llama/Llama-2-7b-chat-hf", max_tokens=max_tokens)
    model_id = "nayohan/closedai-llm"
    # model_id = "mistralai/Mistral-7B-v0.1"
    # model_id = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto", torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=128)
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf


def get_embeddings() -> Embeddings:
    # embeddings = OpenAIEmbeddings(model_name=OPENAI_EMBEDDINGS_ENGINE)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return embeddings


def make_qna_chain(llm: LLM, vector_db: "Chroma", prompt: str = "", **kwargs):
    """Create the QA chain."""

    search_type = "similarity"
    if "search_type" in kwargs:
        search_type = kwargs.pop("search_type")

    # Create retreival QnA Chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs=kwargs),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )

    # chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    return chain

# import os
# from typing import List, TYPE_CHECKING

# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI, ChatOllama

# from langchain.llms import HuggingFaceHub
# from langchain import LLMChain
# from langchain.prompts import PromptTemplate


# from langchain.llms.base import LLM
# from langchain.embeddings.base import Embeddings
# from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# # from qna.constants import (
# #     OPENAI_COMPLETIONS_ENGINE,
# #     OPENAI_EMBEDDINGS_ENGINE,
# # )

# if TYPE_CHECKING:
#     # from langchain.vectorstores.redis import Redis as RedisVDB
#     from langchain.vectorstores import Chroma


# def get_llm(max_tokens=100) -> LLM:
#     # llm = ChatOpenAI(model_name=OPENAI_COMPLETIONS_ENGINE, max_tokens=max_tokens)
#     # llm = ChatOllama(model_name="beomi/llama-2-ko-7b")
#     repo_id = 'mistralai/Mistral-7B-v0.1'
#     llm = HuggingFaceHub(
#         repo_id=repo_id, 
#         model_kwargs={"temperature": 0.2, 
#                     "max_length": 5},
#         huggingfacehub_api_token="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"
#     )
#     return llm


# def get_embeddings() -> Embeddings:
#     embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
#     return embeddings


# def make_qna_chain(llm: LLM, vector_db: "Chroma", prompt: str = "", **kwargs):
#     """Create the QA chain."""

#     search_type = "similarity"
#     if "search_type" in kwargs:
#         search_type = kwargs.pop("search_type")

#     template = """Question: {question} Answer: """
#     prompt = PromptTemplate(template=template, input_variables=["question"])
#     # chain = LLMChain(prompt=prompt, llm=llm)

#     # Create retreival QnA Chain
#     chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         # retriever=vector_db.as_retriever(search_kwargs=kwargs, search_type=search_type),
#         retriever=vector_db.as_retriever(search_kwargs=kwargs),
#         return_source_documents=True,
#         chain_type_kwargs={"prompt": prompt},
#         verbose=True
#     )
#     return chain
