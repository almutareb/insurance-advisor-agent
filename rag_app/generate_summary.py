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
        You are a SEO expert bot. Your task is to craft a meaningful list of 5 keywords to organize documents. 
        The keywords should help us in searching and retrieving the documents later.

        You will only respond with the clear, concise and meaningful 5 of keywords separated by comma. 
          
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
        You are a SEO expert bot. Your task is to craft a meaningful summary to descripe and organize documents. 
        The description should be a meaningful summary of the document's content and help us in searching and retrieving the documents later.

        You will only respond with the clear, concise and meaningful description. 
          
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