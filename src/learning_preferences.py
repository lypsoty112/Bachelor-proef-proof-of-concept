from typing import List, Dict, Any
from langchain.chains.base import Chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from loguru import logger

from src.models.chat import Chat
from src.components.chains.question_extract import question_extract_chain


class Preferences(BaseModel):
    motivation: str = Field(None, description="The person's motivation for learning the subject or skill.")
    skill_level: str = Field(None, description="The person's skill level in this specific subject or skill.")
    study_habits: str = Field(None, description="The person's study habits.")
    study_method: str = Field(None, description="The person's preferred study method. For example, do they "
                                                "prefer visual, auditory, or interactive learning?")
    concentration_level: str = Field(None, description="The person's concentration level.")

    preferred_environment: str = Field(None, description="The person's preferred environment for learning.")
    preferred_feedback: str = Field(None, description="The person's preferred feedback system. "
                                                      "For example, do they prefer positive reinforcement or "
                                                      "constructive criticism? How often do they want "
                                                      "feedback? ...")


class LearningPreferences:
    def __init__(self):

        self._preferences: Dict[str, Any] = Preferences().dict()
        self._chains: Dict[str, Chain] = {
            "question_extract": question_extract_chain(
                object_to_extract=Preferences
            ),
        }

    def run(self, data: List[Chat] | List[Dict]) -> Dict:
        data = self.format(data)
        logger.info(f"Received data: {data}")

        return {
            "role": "assistant",
            "content": 'response'
        }

    @staticmethod
    def format(data: List[Dict] | List[Chat]) -> List[HumanMessage | AIMessage]:
        if isinstance(data[0], dict):
            data = [Chat(**message) for message in data]

        converted_chat_history = []
        for message in data:
            if message.role == "assistant":
                converted_chat_history.append(
                    AIMessage(content=message.content)
                )
            else:
                converted_chat_history.append(
                    HumanMessage(content=message.content)
                )
        return converted_chat_history
