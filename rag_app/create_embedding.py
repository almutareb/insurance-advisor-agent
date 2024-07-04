# embeddings functions
#from langchain_community.vectorstores import FAISS
#from langchain_community.document_loaders import ReadTheDocsLoader
#from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import time
from langchain_core.documents import Document


def create_embeddings(
        docs: list[Document], 
        chunk_size:int = 500, 
        chunk_overlap:int = 50,
        embedding_model: str = "sentence-transformers/multi-qa-mpnet-base-dot-v1", 
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
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    print(f"created a total of {len(chunks)} chunks")

    return embeddings,chunks