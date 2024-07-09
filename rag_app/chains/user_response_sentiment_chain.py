from langchain_core.prompts import PromptTemplate


user_response_sentiment_template = """ 
You will be given a user response to an agent.
=================
{user_reponse} 
====================
You must determine if the user has has their questions answered. 
If the user seems satisfied respond saying "1" or "0" ONLY.

Examples:
================

Example 1
USER: Great Work! 
YOUR RESPONSE: 1
=================

USER: I still need help! 
YOUR RESPONSE: 0
Example 2
================================

USER: I don't understand what you mean 
YOUR RESPONSE: 0
Example 3
================================

USER: That makes sense! 
YOUR RESPONSE: 1
Example 4
================================

  

"""

user_response_sentiment_prompt = PromptTemplate.from_template(user_response_sentiment_template)


# llm_chain = prompt | llms