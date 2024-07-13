# preprocessed vectorstore retrieval
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import zipfile
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from dotenv import load_dotenv
import os
import sys
import logging

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

embeddings = SentenceTransformerEmbeddings(model_name=model_name)

## FAISS
def get_faiss_vs():
    if not os.path.exists(FAISS_INDEX_PATH):
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
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
            
        except Exception as e:
            print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)
        #faissdb = FAISS.load_local(FAISS_INDEX_PATH, embeddings)


## Chroma DB
def get_chroma_vs():
    if not os.path.exists(CHROMA_DIRECTORY):
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
            #chromadb.get()
        except Exception as e:
            print(f"Error during downloading or extracting from S3: {e}", file=sys.stderr)