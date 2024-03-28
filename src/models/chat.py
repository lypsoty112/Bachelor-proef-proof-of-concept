from pydantic import BaseModel, Field
import datetime


class Chat(BaseModel):
    content: str = Field(..., title="Content", description="The content of the chat message.")
    role: str = Field(..., title="Role", description="The role of the chat message.")
    timestamp: datetime.datetime = Field(..., title="Timestamp",
                                         description="The timestamp of the chat message.",)
