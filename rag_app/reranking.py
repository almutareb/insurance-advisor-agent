# from get_db_retriever import get_db_retriever
from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import requests

load_dotenv()


def get_reranked_docs(query:str, 
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

 
if __name__ == "__main__":
    
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    
    path_to_vector_db = Path("..")/'vectorstore/faiss-insurance-agent-500'

    query = "Ich möchte wissen, ob ich meine geriatrische Haustier-Eidechse versichern kann"
    
    top_5_docs = get_reranked_docs(query=query,
                                   path_to_db=path_to_vector_db,
                                   embedding_model=EMBEDDING_MODEL,
                                   hf_api_key=HUGGINGFACEHUB_API_TOKEN,
                                   num_docs=5)
    
    for i, doc in enumerate(top_5_docs):
        print(f"{i}: {doc}\n")