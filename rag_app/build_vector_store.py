# vectorization functions
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from rag_app.create_embedding import create_embeddings
from rag_app.generate_summary import generate_description, generate_keywords
import time
import os
from dotenv import load_dotenv

def build_vector_store(
        docs: list, 
        db_path: str, 
        embedding_model: str, 
        new_db:bool=False, 
        chunk_size:int=500, 
        chunk_overlap:int=50,
        ):
    """

    """

    if db_path is None:
        FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
    else:
        FAISS_INDEX_PATH = db_path

    embeddings,chunks = create_embeddings(docs, chunk_size, chunk_overlap, embedding_model)
    # for chunk in chunks:
    #     keywords=generate_keywords(chunk)
    #     description=generate_description(chunk)
    #     chunk.metadata['keywords']=keywords
    #     chunk.metadata['description']=description

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


# # Path for saving the FAISS index
# FAISS_INDEX_PATH = "./vectorstore/lc-faiss-multi-mpnet-500"

# try:
#     # Stage two: Vectorization of the document chunks
#     model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Model used for embedding

#     # Initialize HuggingFace embeddings with the specified model
#     embeddings = HuggingFaceEmbeddings(model_name=model_name)

#     print(f'Loading chunks into vector store ...')
#     st = time.time()  # Start time for performance measurement
#     # Create a FAISS vector store from the document chunks and save it locally
#     db = FAISS.from_documents(filter_complex_metadata(chunks), embeddings)
#     db.save_local(FAISS_INDEX_PATH)
#     et = time.time() - st  # Calculate time taken for vectorization
#     print(f'Time taken for vectorization and saving: {et} seconds.')
# except Exception as e:
#     print(f"Error during vectorization or FAISS index saving: {e}", file=sys.stderr)

# alternatively download a preparaed vectorized index from S3 and load the index into vectorstore
# Import necessary libraries for AWS S3 interaction, file handling, and FAISS vector stores