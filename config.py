import os
from dotenv import load_dotenv
from rag_app.database.db_handler import DataBaseHandler
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()

SQLITE_FILE_NAME = os.getenv('SOURCES_CACHE')
PERSIST_DIRECTORY = os.getenv('VECTOR_DATABASE_LOCATION')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SECONDARY_LLM_MODEL = os.getenv("SECONDARY_LLM_MODEL")


db = DataBaseHandler()

db.create_all_tables()

SECONDARY_LLM = HuggingFaceEndpoint(
        repo_id=SECONDARY_LLM_MODEL, 
        temperature=0.1,         # Controls randomness in response generation (lower value means less random)
        max_new_tokens=1024,     # Maximum number of new tokens to generate in responses
        repetition_penalty=1.2,  # Penalty for repeating the same words (higher value increases penalty)
        return_full_text=False   # If False, only the newly generated text is returned; if True, the input is included as well
    )