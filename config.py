import os
from dotenv import load_dotenv
from rag_app.database.db_handler import DataBaseHandler
from langchain_huggingface import HuggingFaceEndpoint
# from langchain_huggingface import HuggingFaceHubEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

SQLITE_FILE_NAME = os.getenv('SOURCES_CACHE')
VECTOR_DATABASE_LOCATION = os.getenv('VECTOR_DATABASE_LOCATION')
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
SEVEN_B_LLM_MODEL = os.getenv("SEVEN_B_LLM_MODEL")
BERT_MODEL = os.getenv("BERT_MODEL")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")



# embeddings = HuggingFaceHubEmbeddings(repo_id=EMBEDDING_MODEL)

model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = DataBaseHandler()

db.create_all_tables()

# This model is used for task that a larger model may not need to do
# as of currently we have been getting MODEL OVERLOADED errors
# with huggingface
SEVEN_B_LLM_MODEL = HuggingFaceEndpoint(
        repo_id=SEVEN_B_LLM_MODEL, 
        temperature=0.1,         # Controls randomness in response generation (lower value means less random)
        max_new_tokens=1024,     # Maximum number of new tokens to generate in responses
        repetition_penalty=1.2,  # Penalty for repeating the same words (higher value increases penalty)
        return_full_text=False   # If False, only the newly generated text is returned; if True, the input is included as well
    )