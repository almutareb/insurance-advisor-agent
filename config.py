import os
from dotenv import load_dotenv
from rag_app.database.db_handler import DataBaseHandler

load_dotenv()

sqlite_file_name = os.getenv('SOURCES_CACHE')


db = DataBaseHandler()

db.create_all_tables()

