from langchain.tools import BaseTool, StructuredTool, tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
#from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.vectorstores import Chroma
import ast

import chromadb

from rag_app.utils.utils import (
    parse_list_to_dicts, format_search_results
)
from rag_app.database.db_handler import (
    add_many
)

import os
# from innovation_pathfinder_ai.utils import create_wikipedia_urls_from_text

persist_directory = os.getenv('VECTOR_DATABASE_LOCATION')

@tool
def memory_search(query:str) -> str:
    """Search the memory vector store for existing knowledge and relevent pervious researches. \
        This is your primary source to start your search with checking what you already have learned from the past, before going online."""
    # Since we have more than one collections we should change the name of this tool
    client = chromadb.PersistentClient(
     path=persist_directory,
    )
    
    collection_name = os.getenv('CONVERSATION_COLLECTION_NAME')
    #store using envar
    
    embedding_function = SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
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
    client = chromadb.PersistentClient(
     path=persist_directory,
    )
    
    #collection_name="ArxivPapers"
    #store using envar
    
    embedding_function = SentenceTransformerEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        )
    
    vector_db = Chroma(
    client=client, # client for Chroma
    #collection_name=collection_name,
    embedding_function=embedding_function,
    )
    
    retriever = vector_db.as_retriever(search_kwargs={'k':1})
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
        add_many(parsed_csources)
    else:
        cleaner_sources = search_results
    
    return cleaner_sources.__str__()