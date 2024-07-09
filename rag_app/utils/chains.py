from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceEndpoint

import re


async def get_suggestions_questions(input: str) -> list[str]:
    """Get suggestions questions."""

    llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                    temperature=0.1, 
                                    max_new_tokens=1024,
                                    repetition_penalty=1.2,
                                    return_full_text=False
                )

    prompt_is_farewell_topic_chain = PromptTemplate(
        input_variables=["input"],
        template="Determinate if the '{input}' is related to the topic of farewell and return True or False",
    )
    prompt = PromptTemplate(
        input_variables=["input"],
        template="Create three good suggestions questions about this topic of: {input}. Return the suggestions like a list.",
    )
    is_farewell_topic_chain = LLMChain(llm=llm, prompt=prompt_is_farewell_topic_chain)
    is_farewell_topic_response = await is_farewell_topic_chain.arun(input)
    suggested_responses = []

    if "False" in is_farewell_topic_response:
        chain = LLMChain(llm=llm, prompt=prompt)
        response_chain = await chain.arun(input)
        suggested_responses = re.findall(r"\d+\.\s(.*?\?)", response_chain)
        suggested_responses = suggested_responses[:3]

    return suggested_responses