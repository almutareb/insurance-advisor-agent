# HF libraries
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
# Import things that are needed generically
from langchain.tools.render import render_text_description
import os
from dotenv import load_dotenv
from rag_app.structured_tools.structured_tools import (
    google_search, knowledgeBase_search
)

from langchain.prompts import PromptTemplate
from rag_app.templates.react_json_ger import template_system
# from rag_app.utils import logger

# set_llm_cache(SQLiteCache(database_path=".cache.db"))
# logger = logger.get_console_logger("hf_mixtral_agent")

config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
LLM_MODEL = os.getenv('LLM_MODEL')

# Load the model from the Hugging Face Hub
llm = HuggingFaceEndpoint(repo_id=LLM_MODEL, 
                          temperature=0.1, 
                          max_new_tokens=1024,
                          repetition_penalty=1.2,
                          return_full_text=False
    )


tools = [
    knowledgeBase_search,
    google_search,
    ]

prompt = PromptTemplate.from_template(
    template=template_system
)
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)


# define the agent
chat_model_with_stop = llm.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        #"chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_worker = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    max_iterations=10,       # cap number of iterations
    #max_execution_time=60,  # timout at 60 sec
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    )