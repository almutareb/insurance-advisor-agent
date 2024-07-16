from langchain_core.prompts import PromptTemplate


generate_keywords_template = """ 
You will be given meta data for a chunk text 
=================
{chunk_metadata} 
====================

You will be tasked with creating keywords to help a llm better indentify the correct chunk
to use. Please only return the comma seperate values such that it can easily be parsed.
  

"""

generate_keywords_prompt = PromptTemplate.from_template(generate_keywords_template)

