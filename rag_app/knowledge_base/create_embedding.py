# embeddings functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import time
from langchain_core.documents import Document
from config import EMBEDDING_MODEL

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