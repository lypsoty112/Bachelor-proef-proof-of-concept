from typing import List
from pydantic import BaseModel, Field

from src.models.main import Message


class ChatInput(BaseModel):
    messages: List[Message] = Field(..., description="The messages in the chat conversation.")
    metadata: dict = Field({}, description="The metadata of the chat conversation.")


class ChatOutput(BaseModel):
    response: Message = Field(..., description="The response from the LLM.")
    metadata: dict = Field({}, description="The metadata of the chat conversation.")
