import pytest
from langchain.schema import Document
from rag_app.vector_store_handler.vectorstores import ChromaVectorStore, FAISSVectorStore
# from rag_app.database.init_db import db
from config import EMBEDDING_MODEL, VECTOR_DATABASE_LOCATION
from langchain.embeddings import HuggingFaceEmbeddings  # Or whatever embedding you're using

@pytest.fixture(scope="module")
def embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@pytest.fixture(params=[ChromaVectorStore, FAISSVectorStore])
def vector_store(request, embedding_model, tmp_path):
    store = request.param(embedding_model, persist_directory=str(tmp_path))
    yield store
    # Clean up (if necessary)
    if hasattr(store, 'vectorstore'):
        store.vectorstore.delete_collection()

@pytest.fixture
def sample_documents():
    return [
        Document(page_content="This is a test document about AI."),
        Document(page_content="Another document discussing machine learning."),
        Document(page_content="A third document about natural language processing.")
    ]

def test_create_and_search(vector_store, sample_documents):
    # Create vector store
    vector_store.create_vectorstore(sample_documents)

    # Perform a search
    results = vector_store.similarity_search("AI and machine learning")

    assert len(results) > 0
    assert any("AI" in doc.page_content for doc in results)
    assert any("machine learning" in doc.page_content for doc in results)

def test_save_and_load(vector_store, sample_documents, tmp_path):
    # Create and save vector store
    vector_store.create_vectorstore(sample_documents)
    vector_store.save()

    # Load the vector store
    loaded_store = type(vector_store)(vector_store.embeddings, persist_directory=str(tmp_path))
    loaded_store.load_existing_vectorstore()

    # Perform a search on the loaded store
    results = loaded_store.similarity_search("natural language processing")

    assert len(results) > 0
    assert any("natural language processing" in doc.page_content for doc in results)

def test_update_vectorstore(vector_store, sample_documents):
    # Create initial vector store
    vector_store.create_vectorstore(sample_documents)

    # Add a new document
    new_doc = Document(page_content="A new document about deep learning.")
    vector_store.vectorstore.add_documents([new_doc])

    # Search for the new content
    results = vector_store.similarity_search("deep learning")

    assert len(results) > 0
    assert any("deep learning" in doc.page_content for doc in results)

@pytest.mark.parametrize("query,expected_content", [
    ("AI", "AI"),
    ("machine learning", "machine learning"),
    ("natural language processing", "natural language processing")
])
def test_search_accuracy(vector_store, sample_documents, query, expected_content):
    vector_store.create_vectorstore(sample_documents)
    results = vector_store.similarity_search(query)
    assert any(expected_content in doc.page_content for doc in results)

# def test_database_integration(vector_store, sample_documents):
#     # This test assumes your vector store interacts with the database in some way
#     # You may need to adjust this based on your actual implementation
#     vector_store.create_vectorstore(sample_documents)

#     # Here, you might add some assertions about how the vector store interacts with the database
#     # For example, if you're storing metadata about the documents in the database:
#     for doc in sample_documents:
#         result = db.session.query(YourDocumentModel).filter_by(content=doc.page_content).first()
#         assert result is not None

# Add more integration tests as needed