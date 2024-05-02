from enum import Enum

from pydantic import BaseModel, Field


class NextStep(Enum):
    CONTINUE_COURSE = "continue_course"
    ASK_FOR_MORE_INFORMATION = "ask_for_more_information"
    RESPOND_TO_QUESTION = "respond_to_question"


class ClassifierInput(BaseModel):
    message: str
    chat_history: list


class ClassifierOutput(BaseModel):
    next_step: NextStep = Field(...)
