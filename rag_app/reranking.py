# from get_db_retriever import get_db_retriever
from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import requests
from langchain_community.vectorstores import Chroma


load_dotenv()


def get_reranked_docs_faiss(query:str, 
                      path_to_db:str, 
                      embedding_model:str,
                      hf_api_key:str, 
                      num_docs:int=5) -> list:
    """ Re-ranks the similarity search results and returns top-k highest ranked docs

        Args:
            query (str): The search query
            path_to_db (str): Path to the vectorstore database
            embedding_model (str): Embedding model used in the vector store
            num_docs (int): Number of documents to return
        
        Returns: A list of documents with the highest rank
    """
    assert num_docs <= 10, "num_docs should be less than similarity search results"
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api_key,
                                                   model_name=embedding_model)
    # Load the vectorstore database
    db = FAISS.load_local(folder_path=path_to_db,
                          embeddings=embeddings,
                          allow_dangerous_deserialization=True)
    
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
    assert num_docs <= 10, "num_docs should be less than similarity search results"
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api_key,
                                                   model_name=embedding_model)
    # Load the vectorstore database
    db = Chroma(persist_directory=path_to_db, embedding_function=embeddings)
    
    # Get 10 documents based on similarity search
    sim_docs =  db.similarity_search(query=query, k=10)

    # Add the page_content, description and title together
    passages = [doc.page_content for doc in sim_docs]
    
    # Prepare the payload
    payload = {"inputs": 
               {"source_sentence": query,
	            "sentences": passages}}

    
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    response = requests.post(url=reranking_hf_url, headers=headers, json=payload)
    if response.status_code != 200:
        print('Something went wrong with the response')
        return
    similarity_scores = response.json()
    ranked_results = sorted(zip(sim_docs, passages, similarity_scores), key=lambda x: x[2], reverse=True)
    top_k_results = ranked_results[:num_docs]
    return [doc for doc, _, _ in top_k_results]
    


if __name__ == "__main__":
    
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    EMBEDDING_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

    project_dir = Path().cwd().parent
    path_to_vector_db = str(project_dir/'vectorstore/chroma-zurich-mpnet-1500')

    query = "I'm looking for student insurance"


    re_ranked_docs = get_reranked_docs_chroma(query=query,
                                              path_to_db= path_to_vector_db,
                                              embedding_model=EMBEDDING_MODEL,
                                              hf_api_key=HUGGINGFACEHUB_API_TOKEN)


    print(f"{re_ranked_docs=}")
