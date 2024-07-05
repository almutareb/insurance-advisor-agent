from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
from dotenv import load_dotenv
import os

load_dotenv()

HF_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
model_id=os.getenv('LLM_MODEL')    

LLM = HuggingFaceEndpoint(
repo_id=model_id, 
temperature=0.1, 
max_new_tokens=512,
repetition_penalty=1.2,
return_full_text=False,
huggingfacehub_api_token=HF_API_TOKEN) 

def generate_keywords(document:dict,
                     llm_model:HuggingFaceEndpoint = LLM) -> str:
    """ Generate a meaningful list of meta keywords for the provided document or chunk"""
    
    template = (
        """
        Du bist ein SEO-Experten-Bot. Deine Aufgabe ist es, 5 aussagekräftige Schlüsselwörtern zu erstellen, um Dokumente zu organisieren.
        Die Schlüsselwörter sollen uns später beim Suchen, Filtern und Abrufen der Dokumente helfen.

        Antworte nur mit 5 klaren, prägnanten und aussagekräftigen Schlüsselwörtern, getrennt durch Kommas.
          
        <<<
        Document: {document}
        >>>

        Keywords:
        """
    )

    prompt = PromptTemplate.from_template(template=template)
    
    chain = prompt | llm_model | StrOutputParser()
    result = chain.invoke({'document': document})
    return result.strip()

def generate_description(document:dict,
                     llm_model:HuggingFaceEndpoint = LLM) -> str:
    """ Generate a meaningful document description based on document content """
    
    template = (
        """
        Du bist ein SEO-Experten-Bot. Deine Aufgabe ist es, eine aussagekräftige Zusammenfassung zu erstellen, um Dokumente zu beschreiben und zu organisieren.
        Die Beschreibung sollte eine aussagekräftige Zusammenfassung des Dokumentinhalts sein und uns später beim Suchen und Abrufen der Dokumente helfen.

        Antworte nur mit einer klaren, prägnanten und aussagekräftigen Beschreibung in Deutsch.
          
        <<<
        Document: {document}
        >>>

        Description: 
        """
    )

    prompt = PromptTemplate.from_template(template=template)
    
    chain = prompt | llm_model | StrOutputParser()
    result = chain.invoke({'document': document})
    return result.strip()