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
from config import EMBEDDING_MODEL


def get_db_retriever(vector_db:str=None):
    embeddings = HuggingFaceHubEmbeddings(repo_id=EMBEDDING_MODEL)

    if not vector_db:
        FAISS_INDEX_PATH='./vectorstore/py-faiss-multi-mpnet-500'
    else:
        FAISS_INDEX_PATH=vector_db
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings)

    retriever = db.as_retriever()

    return retriever

