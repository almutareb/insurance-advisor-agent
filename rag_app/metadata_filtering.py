from pathlib import Path
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv(".env")

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


if __name__ == "__main__":

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    folder_path = Path('..') / "vectorstore/faiss-insurance-agent-500"

    print(f'{Path(folder_path).exists() = }')        

    faissdb = FAISS.load_local(folder_path=str(folder_path.resolve()),
                                embeddings=embeddings,
                                allow_dangerous_deserialization=True)
    
    documents = faissdb.get(list(range(5)))

    for doc in documents:
        print(f"Metadata: {doc.metadata}")
