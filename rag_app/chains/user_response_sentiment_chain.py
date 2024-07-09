from langchain_core.prompts import PromptTemplate


user_response_sentiment_template = """ 
Analysiere die folgende Chat Interaktion, und beurteile, ob die Frage, des Nutzers gut beantwortet wurde. Wenn dies der Fall ist, dann biete den Nutzer an, \
Mehr informationen zu den Versicherungsprodukten per Email zuzuschicken und fordere ihn auf seine Email Adresse anzugeben.

Antworte nur mit TRUE oder FALSE. Ohne Erklärungen oder Ergänzung.

Vorheriger Gesprächsverlauf:
<CONVERSATION_HISTORY>
{chat_history}
</CONVERSATION_HISTORY>

BEWERTUNG: 
"""

user_response_sentiment_prompt = PromptTemplate.from_template(user_response_sentiment_template)


# llm_chain = prompt | llms