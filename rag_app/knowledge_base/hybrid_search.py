from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever


def get_hybrid_search_results(query:str,
                              path_to_db:str,
                              embedding_model:str,
                              hf_api_key:str,
                              num_docs:int=5) -> list:
    """ Uses an ensemble retriever of BM25 and FAISS to return k num documents

        Args:
            query (str): The search query
            path_to_db (str): Path to the vectorstore database
            embedding_model (str): Embedding model used in the vector store
            num_docs (int): Number of documents to return
        
        Returns
            List of documents
    
    """
    
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=hf_api_key,
                                                   model_name=embedding_model)
    # Load the vectorstore database
    db = FAISS.load_local(folder_path=path_to_db,
                          embeddings=embeddings,
                          allow_dangerous_deserialization=True)

    all_docs = db.similarity_search("", k=db.index.ntotal)

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = num_docs  # How many results you want

    faiss_retriever = db.as_retriever(search_kwargs={'k': num_docs})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                           weights=[0.5,0.5])
    
    results = ensemble_retriever.invoke(input=query) 
    return results


if __name__ == "__main__":
    query = "Haustierversicherung"
    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    
    path_to_vector_db = Path("..")/'vectorstore/faiss-insurance-agent-500'

    results = get_hybrid_search_results(query=query, 
                                    path_to_db=path_to_vector_db, 
                                    embedding_model=EMBEDDING_MODEL, 
                                    hf_api_key=HUGGINGFACEHUB_API_TOKEN)
    
    for doc in results:
        print(doc)
        print()
