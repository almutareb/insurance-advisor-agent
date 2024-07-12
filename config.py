import os
from dotenv import load_dotenv
from rag_app.database.db_handler import DataBaseHandler

load_dotenv()

SQLITE_FILE_NAME = os.getenv('SOURCES_CACHE')
PERSIST_DIRECTORY = os.getenv('VECTOR_DATABASE_LOCATION')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


db = DataBaseHandler()

db.create_all_tables()

