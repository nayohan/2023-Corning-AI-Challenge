from typing import List

from langchain.schema import Document
from langchain.document_loaders import ArxivLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

file_list = os.listdir("./PaperDB")

def get_arxiv_docs(paper_topic_query, num_docs=10) -> List[Document]:
    # loader = ArxivLoader(
    #     paper_topic_query,
    #     load_max_docs=num_docs,
    #     load_all_available_meta=True
    # )
    loader = DirectoryLoader(
        "PaperDB/", glob="./*.pdf", loader_cls=PyPDFLoader,
        # load_max_docs=len(file_list),
        # load_all_available_meta=True
    )
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 1000,
    chunk_overlap  = 0,
    length_function = len,
    add_start_index = True,
    )
    documents = text_splitter.split_documents(raw_documents)
    return documents

