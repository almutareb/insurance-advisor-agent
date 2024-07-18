from abc import ABC, abstractmethod
from langchain.vectorstores import Chroma, FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import time
from langchain_core.documents import Document
from config import EMBEDDING_MODEL

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.
    
    This class defines the interface for vector stores and implements
    common functionality.
    """

    def __init__(self, embedding_model, persist_directory=None):
        """
        Initialize the BaseVectorStore.

        Args:
            embedding_model: The embedding model to use for vectorizing text.
            persist_directory (str, optional): Directory to persist the vector store.
        """
        self.persist_directory = persist_directory
        self.embeddings = embedding_model
        self.vectorstore = None

    def load_and_process_documents(self, file_path, chunk_size=1000, chunk_overlap=0):
        """
        Load and process documents from a file.

        Args:
            file_path (str): Path to the file to load.
            chunk_size (int): Size of text chunks for processing.
            chunk_overlap (int): Overlap between chunks.

        Returns:
            list: Processed documents.
        """
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_documents(documents)

    @abstractmethod
    def create_vectorstore(self, texts):
        """
        Create a new vector store from the given texts.

        Args:
            texts (list): List of texts to vectorize and store.
        """
        pass

    @abstractmethod
    def load_existing_vectorstore(self):
        """
        Load an existing vector store from the persist directory.
        """
        pass

    def similarity_search(self, query):
        """
        Perform a similarity search on the vector store.

        Args:
            query (str): The query text to search for.

        Returns:
            list: Search results.

        Raises:
            ValueError: If the vector store is not initialized.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore or load_existing_vectorstore first.")
        return self.vectorstore.similarity_search(query)

    @abstractmethod
    def save(self):
        """
        Save the current state of the vector store.
        """
        pass

class ChromaVectorStore(BaseVectorStore):
    """
    Implementation of BaseVectorStore using Chroma as the backend.
    """

    def create_vectorstore(self, texts):
        """
        Create a new Chroma vector store from the given texts.

        Args:
            texts (list): List of texts to vectorize and store.
        """
        self.vectorstore = Chroma.from_documents(
            texts, 
            self.embeddings, 
            persist_directory=self.persist_directory
        )

    def load_existing_vectorstore(self):
        """
        Load an existing Chroma vector store from the persist directory.

        Raises:
            ValueError: If persist_directory is not set.
        """
        if self.persist_directory is not None:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
        else:
            raise ValueError("Persist directory is required for loading Chroma.")

    def save(self):
        """
        Save the current state of the Chroma vector store.

        Raises:
            ValueError: If the vector store is not initialized.
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Nothing to save.")
        self.vectorstore.persist()

class FAISSVectorStore(BaseVectorStore):
    """
    Implementation of BaseVectorStore using FAISS as the backend.
    """

    def create_vectorstore(self, texts):
        """
        Create a new FAISS vector store from the given texts.

        Args:
            texts (list): List of texts to vectorize and store.
        """
        self.vectorstore = FAISS.from_documents(texts, self.embeddings)

    def load_existing_vectorstore(self):
        """
        Load an existing FAISS vector store from the persist directory.

        Raises:
            ValueError: If persist_directory is not set.
        """
        if self.persist_directory:
            self.vectorstore = FAISS.load_local(self.persist_directory, self.embeddings, allow_dangerous_deserialization=True)
        else:
            raise ValueError("Persist directory is required for loading FAISS.")

    def save(self):
        """
        Save the current state of the FAISS vector store.

        Raises:
            ValueError: If the vector store is not initialized.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store not initialized. Nothing to save.")
        self.vectorstore.save_local(self.persist_directory)

# Usage example:
def main():
    """
    Example usage of the vector store classes.
    """
    # Create an embedding model
    embedding_model = OpenAIEmbeddings()
    
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    

    # Using Chroma
    chroma_store = ChromaVectorStore(embedding_model, persist_directory="./chroma_store")
    texts = chroma_store.load_and_process_documents("docs/placeholder.txt")
    chroma_store.create_vectorstore(texts)
    results = chroma_store.similarity_search("Your query here")
    print("Chroma results:", results[0].page_content)
    chroma_store.save()

    # Load existing Chroma store
    existing_chroma = ChromaVectorStore(embedding_model, persist_directory="./chroma_store")
    existing_chroma.load_existing_vectorstore()
    results = existing_chroma.similarity_search("Another query")
    print("Existing Chroma results:", results[0].page_content)

    # Using FAISS
    faiss_store = FAISSVectorStore(embedding_model, persist_directory="./faiss_store")
    texts = faiss_store.load_and_process_documents("path/to/your/file.txt")
    faiss_store.create_vectorstore(texts)
    results = faiss_store.similarity_search("Your query here")
    print("FAISS results:", results[0].page_content)
    faiss_store.save()

    # Load existing FAISS store
    existing_faiss = FAISSVectorStore(embedding_model, persist_directory="./faiss_store")
    existing_faiss.load_existing_vectorstore()
    results = existing_faiss.similarity_search("Another query")
    print("Existing FAISS results:", results[0].page_content)

if __name__ == "__main__":
    main()