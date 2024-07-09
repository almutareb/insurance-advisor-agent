from pydantic import BaseModel
from typing import List, Tuple, Optional

class SourceData(BaseModel):
    human_message: str
    sources: str

class DocumentAddResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class InferResponse(BaseModel):
    output: str
    sources: Optional[List[str]] = []

class BotResponse(BaseModel):
    history: List[Tuple[str, str]]
    response_with_sources: str  # Consolidated response with sources appended