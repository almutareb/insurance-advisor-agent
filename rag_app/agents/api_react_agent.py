# HF libraries
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
# Import things that are needed generically
from langchain.tools.render import render_text_description
from rag_app.schemas.message_schema import (
    IChatResponse,
)
from rag_app.utils.utils import generate_uuid
import os
from dotenv import load_dotenv
from rag_app.utils.adaptive_cards.cards import create_adaptive_card
from rag_app.structured_tools.structured_tools import (
    google_search, knowledgeBase_search, ask_user
)
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from langchain.prompts import PromptTemplate
from rag_app.templates.react_json_with_memory_ger import template_system
from rag_app.utils import logger
from rag_app.utils import utils
# from langchain.globals import set_llm_cache
# from langchain_community.cache import SQLiteCache
from rag_app.utils.callback import (
    CustomAsyncCallbackHandler,
    CustomFinalStreamingStdOutCallbackHandler,
)
from langchain.memory import ConversationBufferMemory
from rag_app.utils.config import settings

#local_cache=settings.LOCAL_CACHE
#set_llm_cache(SQLiteCache(database_path=local_cache))
logger = logger.get_console_logger("hf_mixtral_agent")

config = load_dotenv(".env")
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
GOOGLE_CSE_ID = os.getenv('GOOGLE_CSE_ID')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

router = APIRouter()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@router.websocket("/agent")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    if not settings.HUGGINGFACEHUB_API_TOKEN.startswith("hf_"):
        await websocket.send_json({"error": "HUGGINGFACEHUB_API_TOKEN is not set"})
        return

    while True:
        try:
            data = await websocket.receive_json()
            user_message = data["message"]
            user_message_card = create_adaptive_card(user_message)
            chat_history = []
            resp = IChatResponse(
                sender="you",
                message=user_message_card.to_dict(),
                type="start",
                message_id=generate_uuid(),
                id=generate_uuid(),
            )

            await websocket.send_json(resp.model_dump())
            message_id: str = utils.generate_uuid()
            custom_handler = CustomAsyncCallbackHandler(
                 websocket, message_id=message_id
             )

            # Load the model from the Hugging Face Hub
            llm = HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                    temperature=0.1, 
                                    max_new_tokens=1024,
                                    repetition_penalty=1.2,
                                    return_full_text=False
                )


            tools = [
                knowledgeBase_search,
                google_search,
                ask_user
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
                    "chat_history": lambda x: x["chat_history"],
                }
                | prompt
                | chat_model_with_stop
                | ReActJsonSingleInputOutputParser()
            )

            # instantiate AgentExecutor
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True,
                max_iterations=10,       # cap number of iterations
                #max_execution_time=60,  # timout at 60 sec
                return_intermediate_steps=True,
                handle_parsing_errors=True,
                #memory=memory
                )
            
            await agent_executor.arun(input=user_message, chat_history=chat_history, callbacks=[custom_handler])
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
            break