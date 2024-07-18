from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import csv
from pathlib import Path
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()



def get_chunks(website_docs:str, sources:list[tuple[str,str]]) -> list:
    """ Returns chunks from a collection of pdfs

        Args:
            website_docs (str): The path to the folder with the pdfs
            sources (str): The path to the csv with the pdf file name and sources

        Returns: A list of Document objects with the chunks
    """

    website_doc_dir = Path(website_docs)
    assert website_doc_dir.is_dir(), "Cannot find folder website_doc_dir"

    pdf_files_loc = list(website_doc_dir.glob('*.pdf'))
    all_chunks = []  # This will store all chunks from all PDFs

    for file in pdf_files_loc:
        for source in sources:
            if file.name == source[0]:  # Using file.name instead of file.parts[-1]
                # print(f"Processing {file.name}")
                
                loader = PyPDFLoader(str(file))  # Convert Path to string
                pages = loader.load_and_split()
                
                # Create a text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )

                # Split the pages into chunks
                chunks = text_splitter.split_documents(pages)
                
                # Add source to each chunk's metadata
                for chunk in chunks:
                    chunk.metadata['source'] = source[1]
                
                # Add these chunks to our list of all chunks
                all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":

    with open(file='website_docs/pdf_and_source.csv', mode='r') as f:
        csv_reader = csv.reader(f)
        pdf_source = [(row[0], row[1].strip()) for row in csv_reader]
 
    chunks = get_chunks(website_docs='website_docs', sources=pdf_source)

    HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HUGGINGFACEHUB_API_TOKEN,
    model_name="thenlper/gte-small")

    # Create a database from chunks
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local(folder_path="vectorstore_db")

    print("Done saving vectorstore")
    
