# vectorization functions
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from rag_app.knowledge_base.create_embedding import create_embeddings
from rag_app.utils.generate_summary import generate_description, generate_keywords

import time
import os

from config import FAISS_INDEX_PATH

def build_vector_store(
        docs: list, 
        embedding_model: str, 
        new_db:bool=False, 
        chunk_size:int=500, 
        chunk_overlap:int=50,
        ):
    """

    """

    embeddings,chunks = create_embeddings(
        docs, 
        chunk_size, 
        chunk_overlap, 
        embedding_model
        )

    #load chunks into vector store
    print(f'Loading chunks into faiss vector store ...')
    
    st = time.time()
    if new_db:
        db_faiss = FAISS.from_documents(chunks, embeddings)
        bm25_retriever = BM25Retriever.from_documents(chunks)
    else:
        db_faiss = FAISS.add_documents(chunks, embeddings)
        bm25_retriever = BM25Retriever.add_documents(chunks)
        
    db_faiss.save_local(FAISS_INDEX_PATH)
    et = time.time() - st
    print(f'Time taken: {et} seconds.')

    print(f'Loading chunks into chroma vector store ...')
    
    st = time.time()
    persist_directory='./vectorstore/chroma-insurance-agent-1500'
    db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    et = time.time() - st
    
    print(f'Time taken: {et} seconds.')
    result = f"built vectore store at {FAISS_INDEX_PATH}"
    return result

