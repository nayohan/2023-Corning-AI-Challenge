from typing import List
from langchain.schema import Document
from langchain.text_splitter import *
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from prompt import basic_prompt
from tqdm import tqdm

import pandas as pd
import openai
import time
import argparse

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_path_or_name', type=str, required=True)
    p.add_argument('--input_path', type=str, required=True)
    p.add_argument('--output_path', type=str, required=True)
    
    p.add_argument('--output_dir_retriever', type=str, required=True)
    p.add_argument('--output_dir_generator', type=str, required=True)
    
    p.add_argument('--chunk_size', type=int, default=256)
    p.add_argument('--chunk_overlap', type=int, default=0)
    p.add_argument('--chunk_mode', type=str, default='recursive', choices=['charater', 'recursive', 'markdown'])
    
    p.add_argument('--extension', type=str, default='pdf', choices=['pdf', 'docx', 'all'])
    
    p.add_argument('--doc_length', type=int, default=200)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--openai_api_key', type=str, required=True)
    
    args = p.parse_args()
    return args

class Chunking():
    def __init__(self, file_path, chunk_size=256, chunk_overlap=0, mode='recursive', extension='pdf'):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.mode = mode
        self.extension = extension
        
    def _pdf_load(self):
        return DirectoryLoader(self.file_path, glob="./*.pdf", loader_cls=PyPDFLoader).load()

    def _docx_load(self):
        return DirectoryLoader(self.file_path, glob="./*.docx", loader_cls=UnstructuredWordDocumentLoader).load()

    def _chunk_docs(self, origin_docs):
        origin_docs_ = [doc.page_content for doc in origin_docs]
        if self.mode == 'charater':
            text_splitter = CharacterTextSplitter(separator="\n\n", 
                                                  chunk_size=self.chunk_size, 
                                                  chunk_overlap=self.chunk_overlap)
            docs = text_splitter.create_documents(str(origin_docs_))
            docs = [doc.page_content for doc in docs]
        
        elif self.mode == 'recursive':
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, 
                                                           chunk_overlap=self.chunk_overlap, 
                                                           length_function=len, 
                                                           add_start_index=True)
            docs = text_splitter.split_documents(origin_docs)
            docs = [doc.page_content for doc in docs]
        
        elif self.mode == 'spacy':
            text_splitter = SpacyTextSplitter()
            documents = text_splitter.split_documents(raw_documents)
            documents = [doc.page_content for doc in documents]
        
        elif self.mode == 'markdown':
            text_splitter = MarkdownTextSplitter(chunk_size=self.chunk_size, 
                                                 chunk_overlap=self.chunk_overlap)
            docs = text_splitter.create_documents(origin_docs_)
            docs = [doc.page_content for doc in docs]
        
        return docs
    
    def get_docs(self) -> List[Document]:
        if self.extension == 'pdf':
            pdf_docs = self._pdf_load()
            return self._chunk_docs(pdf_docs)
        
        elif self.extension == 'docx':
            docx_docs = self._docx_load()
            return self._chunk_docs(docx_docs)
        
        elif self.extension == 'all':
            pdf_docs = self._pdf_load()
            docx_docs = self._docx_load()
            return self._chunk_docs(pdf_docs), self._chunk_docs(docx_docs)


def main(args):
    openai.api_key = args.openai_api_key
    if args.extension == 'all': 
        chunked_pdf, chunked_docx = Chunking(
                        file_path=args.input_path, 
                        chunk_size=args.chunk_size, 
                        chunk_overlap=args.chunk_overlap, 
                        mode=args.chunk_mode, 
                        extension=args.extension
                        ).get_docs()
    else:
        chunked_docs = Chunking(
                            file_path=args.input_path, 
                            chunk_size=args.chunk_size, 
                            chunk_overlap=args.chunk_overlap, 
                            mode=args.chunk_mode, 
                            extension=args.extension
                            ).get_docs()
        chunked_docs = [doc for doc in chunked_docs if len(doc) >= args.doc_length]        
    idx = 0
    context, query, answer = [], [], []
    
    for i in tqdm(range(len(chunked_docs))):
        prompt = basic_prompt()
        response = openai.Completion.create(
                            model=args.model_path_or_name,
                            prompt=prompt.format(chunked_docs[i]),
                            max_tokens = args.max_length
                            )
        try:
            output=response.choices[0].text.replace("\n질문: ","").replace("\n질문 : ","").replace("\nQuestion: ","").replace("\nQuestion : ","").replace("\nQ: ","").replace("\n답변: ",">>").replace("\n답변 : ",">>").replace("\nAnswer: ",">>").replace("\nAnswer : ",">>").replace("\nA: ",">>").split(">>")
            answer.append(output[1].replace('\n', '').replace('\t','').replace('Answer:',''))
            query.append(output[0].replace('\n', '').replace('\t','').replace('Question:',''))
            context.append(chunked_docs[i].replace('\n', '').replace('\t',''))
            print('query:', query[idx])
            print('answer:', answer[idx])
            idx += 1
        except IndexError:
            pass
            
        data = pd.DataFrame({'context': context, 'query': query, 'answer': answer})
        data.to_json(args.output_path, orient="records", lines=True, force_ascii = False) 
    
    data4generator = data
    data4retriever = data[['context', 'query']]
    data4generator.to_json(args.output_dir_generator, orient="records", lines=True, force_ascii = False)
    data4retriever.to_json(args.output_dir_retriever, orient="records", lines=True, force_ascii = False)
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)
