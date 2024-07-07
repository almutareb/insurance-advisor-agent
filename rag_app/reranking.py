# from get_db_retriever import get_db_retriever
from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HUGGINGFACEHUB_API_TOKEN, 
                                               model_name=EMBEDDING_MODEL)

path_to_vector_db = Path("..")/'vectorstore'/'faiss-insurance-agent-500'

db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

# retreiver = get_db_retriever(vector_db=Path("..")/)

if __name__ == "__main__":
    print(path_to_vector_db.exists())