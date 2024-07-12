from langchain.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
from rag_app.utils.utils import (
    parse_list_to_dicts, format_search_results
)
import chromadb
import os
from config import db, PERSIST_DIRECTORY, EMBEDDING_MODEL


@tool
def memory_search(query:str) -> str:
    """Search the memory vector store for existing knowledge and relevent pervious researches. \
        This is your primary source to start your search with checking what you already have learned from the past, before going online."""
    # Since we have more than one collections we should change the name of this tool
    client = chromadb.PersistentClient(
     path=PERSIST_DIRECTORY,
    )
    
    collection_name = os.getenv('CONVERSATION_COLLECTION_NAME')
    #store using envar
    
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL,
        )
    
    vector_db = Chroma(
    client=client, # client for Chroma
    collection_name=collection_name,
    embedding_function=embedding_function,
    )
    
    retriever = vector_db.as_retriever()
    docs = retriever.invoke(query)
    
    
    return docs.__str__()


@tool
def knowledgeBase_search(query:str) -> str:
    """Suche die interne Datenbank nach passenden Versicherungsprodukten und Informationen zu den Versicherungen"""
    # Since we have more than one collections we should change the name of this tool
    # client = chromadb.PersistentClient(
    #  path=persist_directory,
    # )
    
    #collection_name="ArxivPapers"
    #store using envar
    
    embedding_function = SentenceTransformerEmbeddings(
        model_name=EMBEDDING_MODEL
        )
    
    # vector_db = Chroma(
    # client=client, # client for Chroma
    # #collection_name=collection_name,
    # embedding_function=embedding_function,
    # )
    vector_db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_function)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={'k':5, 'fetch_k':10})
    # This is deprecated, changed to invoke
    # LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` was deprecated in langchain-core 0.1.46 and will be removed in 0.3.0. Use invoke instead.
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc)
    
    return docs.__str__()


@tool
def google_search(query: str) -> str:
    """Verbessere die Ergebnisse durch eine Suche über die Webseite der Versicherung. Erstelle eine neue Suchanfrage, um die Erfolgschancen zu verbesseren."""
    
    websearch = GoogleSearchAPIWrapper()
    search_results:dict = websearch.results(query, 3)
    print(search_results)
    if len(search_results)>1:
        cleaner_sources =format_search_results(search_results)
        parsed_csources = parse_list_to_dicts(cleaner_sources)
        db.add_many(parsed_csources)
    else:
        cleaner_sources = search_results
    
    return cleaner_sources.__str__()
