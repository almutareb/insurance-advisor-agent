from langchain_core.documents import Document
from chains import generate_document_summary_prompt
# embeddings functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import time
from langchain_core.language_models import BaseChatModel
from langchain.retrievers import VectorStoreRetriever
from langchain_core.vectorstores import VectorStoreRetriever
# vectorization functions
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings


from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
import requests

from rag_app.knowledge_base.utils import create_embeddings
from rag_app.utils.generate_summary import generate_description, generate_keywords
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, SEVEN_B_LLM_MODEL

def create_embeddings(
        docs: list[Document], 
        chunk_size:int = 500, 
        chunk_overlap:int = 50,
        ):
    """given a sequence of `Document` objects this fucntion will
    generate embeddings for it.
    
    ## argument
    :params docs (list[Document]) -> list of `list[Document]`
    :params chunk_size (int) -> chunk size in which documents are chunks, defaults to 500
    :params chunk_overlap (int) -> the amount of token that will be overlapped between chunks, defaults to 50
    :params embedding_model (str) -> the huggingspace model that will embed the documents 
    ## Return
    Tuple of embedding and chunks
    """
    
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap,
        length_function = len,
    )

    # Stage one: read all the docs, split them into chunks.
    st = time.time()
    print('Loading documents and creating chunks ...')

    # Split each document into chunks using the configured text splitter
    chunks = text_splitter.create_documents([doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs])
    et = time.time() - st
    print(f'Time taken to chunk {len(docs)} documents: {et} seconds.')

    #Stage two: embed the docs.
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"created a total of {len(chunks)} chunks")

    return embeddings,chunks


def generate_document_summaries(
        docs: list[Document],
        llm:BaseChatModel= SEVEN_B_LLM_MODEL,
    ) -> list[Document]:
    """
    Generates summaries for a list of Document objects and updates their metadata with the summaries.

    Args:
        docs (List[Document]): A list of Document objects to generate summaries for.

    Returns:
        List[Document]: A new list of Document objects with updated metadata containing the summaries.

    Example:
        docs = [Document(metadata={"title": "Doc1"}), Document(metadata={"title": "Doc2"})]
        updated_docs = generate_document_summaries(docs)
        for doc in updated_docs:
            print(doc.metadata["summary"])

    """
    
    new_docs = docs.copy()
    
    for doc in new_docs:
        
        genrate_summary_chain = generate_document_summary_prompt | llm
        summary = genrate_summary_chain.invoke(
            {"document":str(doc.metadata)}
        )        
        
        doc.metadata.update(
            {"summary":summary}
        )
    
    return new_docs


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

    