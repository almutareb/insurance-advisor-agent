from pydantic import BaseModel, validator
from typing import List, Tuple, Optional
from rag_app.utils.utils import generate_uuid
from typing import Any

class InferRequest(BaseModel):
    question: str
    history: List[Tuple[str, str]]  

class BotRequest(BaseModel):
    history: List[Tuple[str, str]]

class IChatResponse(BaseModel):
    """Chat response schema."""

    id: str
    message_id: str
    sender: str
    message: Any
    type: str
    suggested_responses: list[str] = []

    @validator("id", "message_id", pre=True, allow_reuse=True)
    def check_ids(cls, v):
        if v == "" or v is None:
            return generate_uuid()
        return v

    # @validator("sender")
    # def sender_must_be_bot_or_you(cls, v):
    #     if v not in ["bot", "you"]:
    #         raise ValueError("sender must be bot or you")
    #     return v

    # @validator("type")
    # def validate_message_type(cls, v):
    #     if v not in ["start", "stream", "end", "error", "info"]:
    #         raise ValueError("type must be start, stream or end")
    #     return v