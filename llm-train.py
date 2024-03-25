import ollama
from PyPDF2 import PdfReader 
import chromadb
from chromadb.utils import embedding_functions
chroma_client = chromadb.Client()
import pypdf

import os
from langchain_community.vectorstores import Chroma
# from langchain.embeddings import OpenAIEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain.chains import RetrievalQA

from datetime import datetime as time


t1 = time.now() 

# Ollama embeddings
embeddings_open = OllamaEmbeddings(model="mixtral")

llm_open = Ollama(model="mixtral",
                    #model='Llama2',
                    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))



# Print number of txt files in directory
# loader = DirectoryLoader('.../data/langchain_doc_small', glob="./*.txt")

# load pdfs from directory and print number of pdfs
# loader = PyPDFLoader('fdafood.pdf')

# # load another file directly
# # loader = DirectoryLoader('/your/path/to/file.txt')
# doc = loader.load()
# print(doc[0])


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# texts = text_splitter.split_documents(doc)

# print(len(texts))

# persist_directory = "vdb_langchain_doc_small"

# vectordb = Chroma.from_documents(texts,
#                                  embeddings_open,
#                                  persist_directory=persist_directory)

# vectordb.persist()

# vectordb = None

# t2 = time.now()

# print(f"done, {t2-t1}")


persist_directory = "vdb_langchain_doc_small"

vectordb = Chroma(persist_directory=persist_directory,
                        embedding_function= embeddings_open)

retriever =vectordb.as_retriever()

docs = retriever.get_relevant_documents("What is this document about?")

print(docs)

# Read PDF and get text    
# reader = PdfReader('fdafood.pdf')
# page_ids=[]
# page_texts=[]

# for i,p in enumerate(reader.pages):
#     page_ids.append(str(i))
#     page_texts.append(p.extract_text())
    
t2 = time.now()

print(f"done, {t2-t1}")

    
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])    
    
    
qa_chain = RetrievalQA.from_chain_type(llm=llm_open,
                                chain_type="stuff",
                                retriever=retriever,
                                return_source_documents=True,
                                verbose=True)

query = "What expert elicitation did ERG conduct to improve the current state of food safety?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

    
# # response = ollama.chat(model='mixtral', messages=[
# #   {
# #     'role': 'user',
# #     'content': 'Why is the sky blue?',
# #   },
# # ])
# # print(response['message']['content'])
