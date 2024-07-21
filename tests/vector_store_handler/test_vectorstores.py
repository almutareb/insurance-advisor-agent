import unittest
from unittest.mock import MagicMock, patch
# from langchain.embeddings import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
from langchain_core.documents import Document

# Update the import to reflect your project structure
from rag_app.vector_store_handler.vectorstores import BaseVectorStore, ChromaVectorStore, FAISSVectorStore

class TestBaseVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding_model = MagicMock(spec=HuggingFaceEmbeddings)
        self.base_store = BaseVectorStore(self.embedding_model, "test_dir")

    def test_init(self):
        self.assertEqual(self.base_store.persist_directory, "test_dir")
        self.assertEqual(self.base_store.embeddings, self.embedding_model)
        self.assertIsNone(self.base_store.vectorstore)

    @patch('rag_app.vector_store_handler.vectorstores.TextLoader')
    @patch('rag_app.vector_store_handler.vectorstores.CharacterTextSplitter')
    def test_load_and_process_documents(self, mock_splitter, mock_loader):
        mock_loader.return_value.load.return_value = ["doc1", "doc2"]
        mock_splitter.return_value.split_documents.return_value = ["split1", "split2"]

        result = self.base_store.load_and_process_documents("test.txt")

        mock_loader.assert_called_once_with("test.txt")
        mock_splitter.assert_called_once_with(chunk_size=1000, chunk_overlap=0)
        self.assertEqual(result, ["split1", "split2"])

    def test_similarity_search_not_initialized(self):
        with self.assertRaises(ValueError):
            self.base_store.similarity_search("query")

class TestChromaVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding_model = MagicMock(spec=HuggingFaceEmbeddings)
        self.chroma_store = ChromaVectorStore(self.embedding_model, "test_dir")

    @patch('rag_app.vector_store_handler.vectorstores.Chroma')
    def test_create_vectorstore(self, mock_chroma):
        texts = [Document(page_content="test")]
        self.chroma_store.create_vectorstore(texts)
        mock_chroma.from_documents.assert_called_once_with(
            texts, 
            self.embedding_model, 
            persist_directory="test_dir"
        )

    @patch('rag_app.vector_store_handler.vectorstores.Chroma')
    def test_load_existing_vectorstore(self, mock_chroma):
        self.chroma_store.load_existing_vectorstore()
        mock_chroma.assert_called_once_with(
            persist_directory="test_dir",
            embedding_function=self.embedding_model
        )

    def test_save(self):
        self.chroma_store.vectorstore = MagicMock()
        self.chroma_store.save()
        self.chroma_store.vectorstore.persist.assert_called_once()

class TestFAISSVectorStore(unittest.TestCase):
    def setUp(self):
        self.embedding_model = MagicMock(spec=HuggingFaceEmbeddings)
        self.faiss_store = FAISSVectorStore(self.embedding_model, "test_dir")

    @patch('rag_app.vector_store_handler.vectorstores.FAISS')
    def test_create_vectorstore(self, mock_faiss):
        texts = [Document(page_content="test")]
        self.faiss_store.create_vectorstore(texts)
        mock_faiss.from_documents.assert_called_once_with(texts, self.embedding_model)

    @patch('rag_app.vector_store_handler.vectorstores.FAISS')
    def test_load_existing_vectorstore(self, mock_faiss):
        self.faiss_store.load_existing_vectorstore()
        mock_faiss.load_local.assert_called_once_with("test_dir", self.embedding_model)

    @patch('rag_app.vector_store_handler.vectorstores.FAISS')
    def test_save(self, mock_faiss):
        self.faiss_store.vectorstore = MagicMock()
        self.faiss_store.save()
        self.faiss_store.vectorstore.save_local.assert_called_once_with("test_dir")

if __name__ == '__main__':
    unittest.main()