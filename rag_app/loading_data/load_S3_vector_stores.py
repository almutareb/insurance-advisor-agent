# preprocessed vectorstore retrieval
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import sys
import logging
from pathlib import Path

# Load environment variables from a .env file
config = load_dotenv(".env")

# Retrieve the Hugging Face API token from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
S3_LOCATION = os.getenv("S3_LOCATION")
FAISS_VS_NAME = os.getenv("FAISS_VS_NAME")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
CHROMA_DIRECTORY = os.getenv("CHROMA_DIRECTORY")
CHROMA_VS_NAME = os.getenv("CHROMA_VS_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

model_name = EMBEDDING_MODEL
#model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
#    model_kwargs=model_kwargs
    )

## FAISS
def get_faiss_vs():
    # Initialize an S3 client with unsigned configuration for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Define the destination for the downloaded file
    VS_DESTINATION = FAISS_INDEX_PATH + ".zip"
    
    try:
        # Download the pre-prepared vectorized index from the S3 bucket
        print("Downloading the pre-prepared FAISS vectorized index from S3...")
        s3.download_file(S3_LOCATION, FAISS_VS_NAME, VS_DESTINATION)

        # Extract the downloaded zip file
        with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
            zip_ref.extractall('./vectorstore/')
        print("Download and extraction completed.")
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings,allow_dangerous_deserialization=True)
        
    except Exception as e:
        print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)
    # faissdb = FAISS.load_local(FAISS_INDEX_PATH, embeddings)


def get_faiss_vs_from_s3(s3_loc:str, 
                         s3_vs_name:str,
                         vs_dir:str='vectorstore') -> None:
    """ Download the FAISS vector store from S3 bucket

        Args:
            s3_loc (str): Name of the S3 bucket
            s3_vs_name (str): Name of the file to be downloaded
            vs_dir (str): The name of the directory where the file is to be saved
    """
    # Initialize an S3 client with unsigned configuration for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    # Destination folder
    vs_dir_path = Path("..") / vs_dir
    assert vs_dir_path.is_dir(), "Cannot find vs_dir folder"
    try:
        vs_destination = Path("..") / vs_dir / "faiss-insurance-agent-500.zip"
        s3.download_file(s3_loc, s3_vs_name, vs_destination)
        # Extract the downloaded zip file
        with zipfile.ZipFile(file=vs_destination, mode='r') as zip_ref:
            zip_ref.extractall(path=vs_dir_path.as_posix())
    except Exception as e:
        print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)


## Chroma DB
def get_chroma_vs():
    # Initialize an S3 client with unsigned configuration for public access
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    VS_DESTINATION = CHROMA_DIRECTORY+".zip"
    try:
        # Download the pre-prepared vectorized index from the S3 bucket
        print("Downloading the pre-prepared chroma vectorstore from S3...")
        s3.download_file(S3_LOCATION, CHROMA_VS_NAME, VS_DESTINATION)
        with zipfile.ZipFile(VS_DESTINATION, 'r') as zip_ref:
            zip_ref.extractall('./vectorstore/')
        print("Download and extraction completed.")
        chromadb = Chroma(persist_directory=CHROMA_DIRECTORY, embedding_function=embeddings)
        chromadb.get()
    except Exception as e:
        print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)


if __name__ == "__main__":
    # get_faiss_vs_from_s3(s3_loc=S3_LOCATION, s3_vs_name=FAISS_VS_NAME)
    pass
    