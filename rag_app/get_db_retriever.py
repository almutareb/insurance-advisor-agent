# retriever and qa_chain function

# HF libraries
from langchain.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceHubEmbeddings
# vectorestore
from langchain_community.vectorstores import FAISS
# retrieval chain
from langchain.chains import RetrievalQA
# prompt template
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from config import EMBEDDING_MODEL, VECTOR_DATABASE_LOCATION


def get_db_retriever():
    """
    Creates and returns a retriever object based on a FAISS vector database.

    This function initializes an embedding model and loads a pre-existing FAISS
    vector database from a local location. It then creates a retriever from this
    database.

    Returns:
    --------
    retriever : langchain.vectorstores.FAISS.VectorStoreRetriever
        A retriever object that can be used to fetch relevant documents from the
        vector database.

    Global Variables Used:
    ----------------------
    EMBEDDING_MODEL : str
        The identifier for the Hugging Face Hub embedding model to be used.
    VECTOR_DATABASE_LOCATION : str
        The local path where the FAISS vector database is stored.

    Dependencies:
    -------------
    - langchain_huggingface.HuggingFaceHubEmbeddings
    - langchain_community.vectorstores.FAISS

    Note:
    -----
    This function assumes that a FAISS vector database has already been created
    and saved at the location specified by VECTOR_DATABASE_LOCATION.
    """

    # Initialize the embedding model
    embeddings = HuggingFaceHubEmbeddings(repo_id=EMBEDDING_MODEL)
    
    # Load the FAISS vector database from the local storage
    db = FAISS.load_local(
        VECTOR_DATABASE_LOCATION,
        embeddings,
    )

    # Create and return a retriever from the loaded database
    retriever = db.as_retriever()
    
    return retriever

