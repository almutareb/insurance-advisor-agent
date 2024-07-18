from langchain_core.documents import Document
from chains import generate_document_summary_prompt
from config import SEVEN_B_LLM_MODEL
# embeddings functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import time
from langchain_core.documents import Document
from config import EMBEDDING_MODEL
from langchain.retrievers import VectorStoreRetriever
from langchain_core.vectorstores import VectorStoreRetriever
# vectorization functions
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever

from rag_app.knowledge_base.utils import create_embeddings
from rag_app.utils.generate_summary import generate_description, generate_keywords

import time
import os

from config import FAISS_INDEX_PATH

from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import requests
from langchain_community.vectorstores import Chroma



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
        docs: list[Document]
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
        
        genrate_summary_chain = generate_document_summary_prompt | SEVEN_B_LLM_MODEL
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

def get_reranked_docs_faiss(
    query:str, 
    path_to_db:str, 
    embedding_model:str,
    hf_api_key:str, 
    num_docs:int=5
    ) -> list:
    """ Re-ranks the similarity search results and returns top-k highest ranked docs

    Args:
        query (str): The search query
        path_to_db (str): Path to the vectorstore database
        embedding_model (str): Embedding model used in the vector store
        num_docs (int): Number of documents to return
    
    Returns: A list of documents with the highest rank
    """
    assert num_docs <= 10, "num_docs should be less than similarity search results"
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_api_key,
        model_name=embedding_model
        )
    
    # Load the vectorstore database
    db = FAISS.load_local(
        folder_path=path_to_db,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
        )
    
    # Get 10 documents based on similarity search
    docs =  db.similarity_search(query=query, k=10)

    # Add the page_content, description and title together
    passages = [doc.page_content + "\n" + doc.metadata.get('title', "") +"\n"+ doc.metadata.get('description', "") 
                for doc in docs]
    
    # Prepare the payload
    inputs = [{"text": query, "text_pair": passage} for passage in passages]

    API_URL = "https://api-inference.huggingface.co/models/deepset/gbert-base-germandpr-reranking"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(API_URL, headers=headers, json=inputs)
    scores = response.json()
    
    try:
        relevance_scores = [item[1]['score'] for item in scores]
    except ValueError as e:
        print('Could not get the relevance_scores -> something might be wrong with the json output')
        return 
    
    if relevance_scores:
        ranked_results = sorted(zip(docs, passages, relevance_scores), key=lambda x: x[2], reverse=True)
        top_k_results = ranked_results[:num_docs]
        return [doc for doc, _, _ in top_k_results]
    
    
def get_reranked_docs_chroma(query:str, 
                      path_to_db:str, 
                      embedding_model:str,
                      hf_api_key:str,
                      reranking_hf_url:str = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2", 
                      num_docs:int=5) -> list:
    """ Re-ranks the similarity search results and returns top-k highest ranked docs

        Args:
            query (str): The search query
            path_to_db (str): Path to the vectorstore database
            embedding_model (str): Embedding model used in the vector store
            num_docs (int): Number of documents to return
        
        Returns: A list of documents with the highest rank
    """
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api_key,
                                                   model_name=embedding_model)
    # Load the vectorstore database
    db = Chroma(persist_directory=path_to_db, embedding_function=embeddings)
    
    # Get k documents based on similarity search
    sim_docs =  db.similarity_search(query=query, k=10)

    passages = [doc.page_content for doc in sim_docs]
    
    # Prepare the payload
    payload = {"inputs": 
               {"source_sentence": query,
	            "sentences": passages}}
    
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(url=reranking_hf_url, headers=headers, json=payload)
    print(f'{response = }')
    if response.status_code != 200:
        print('Something went wrong with the response')
        return
    
    similarity_scores = response.json()
    ranked_results = sorted(zip(sim_docs, passages, similarity_scores), key=lambda x: x[2], reverse=True)
    top_k_results = ranked_results[:num_docs]
    return [doc for doc, _, _ in top_k_results]