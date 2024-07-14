from langchain_core.prompts import PromptTemplate


query_rewritting_template = """ 
You will be given a query from a user
=================
{user_query} 
====================

You must improve the query to optimize the result
  

"""

query_rewritting_prompt = PromptTemplate.from_template(query_rewritting_template)

