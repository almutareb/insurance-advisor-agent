from langchain_core.prompts import PromptTemplate


generate_document_summary_template = """ 
You will be given a document object
=================
{document} 
====================
You must generate a summary 


"""

generate_document_summary_prompt = PromptTemplate.from_template(generate_document_summary_template)
