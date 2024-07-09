from fastapi import APIRouter
from rag_app.agents import (
    api_react_agent,
)

api_router = APIRouter()

api_router.include_router(api_react_agent.router, prefix="/chat", tags=["chat"])