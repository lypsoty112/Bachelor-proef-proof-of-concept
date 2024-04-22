from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.chains.baseChain import BaseChain
from src.chains.simpleTextTaskChain import SimpleTextTaskChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.chat import ChatInput, ChatOutput
from src.models.chains.simpleTextTask import SimpleTextTaskInput
from src.models.main import Message

PROMPT = """
As an AI, you are tasked with teaching a specific HR course to a user. The user has provided their preferred learning methods and has also outlined their learning goals for this course. \
Your task is to explain the course content in a way that aligns with both the user's learning goals and their preferred learning methods. The explanation should be engaging, easy to understand, \
and tailored to the user's needs. You should also use a conversational approach to make the learning process interactive. Format your messages in markdown.

Here's the input: 
**Course**: 
``
{course}
``

**Learning Goals**:
{learning_goals}

**User Preferences**:
{user_preferences}
"""


class TeacherChain(BaseChain):
    def __init__(self, contents: List[Dict[str, Any]], user_preferences: List[str]):
        super().__init__()
        self._course = contents
        self._user_preferences = user_preferences
        self._llm = OpenAILLM(chat=True)
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Teacher Chain"
        self.prompt = PROMPT
        self._chunk_idx = 0
        self._inner_chunk_idx = 0
        self._chains = {"simple_task": SimpleTextTaskChain()}

    def build(self) -> None:
        self._chains["simple_task"].build()
        built_llm = self._llm.build()

        messages = [("system", self.prompt), MessagesPlaceholder(variable_name="messages")]

        prompt = ChatPromptTemplate.from_messages(messages)

        self._chain = prompt | built_llm

    def starting_message(self) -> ChatOutput:
        response = self._chains["simple_task"].run(SimpleTextTaskInput(text=self._course[self._chunk_idx]["chunk"]["page_content"],
                                                                       task="Write an opening message as a teacher called Teach. The message is meant for 1 student and should be engaging and "
                                                                            "informative. End the"
                                                                            "message with this sentence: 'Are you ready to start?'. Format your message in markdown."))
        return ChatOutput(response=Message(role="ai", content=response.output), metadata={"metadata": "metadata"})

    def run(self, data: ChatInput) -> ChatOutput:
        pre_data = self.pre_run(data)
        self._logger.info(f"Running the chain...")
        response = self._chain.invoke(
            input={"messages": pre_data["messages"], "course": pre_data["course"], "learning_goals": pre_data["learning_goals"], "user_preferences": pre_data["user_preferences"]})
        return self.post_run(response)

    def pre_run(self, data: ChatInput) -> dict:
        return {"messages": [(m.role, m.content) for m in data.messages[-10:]], "course": self._course[self._chunk_idx]["chunk"]["page_content"],
                "learning_goals": self._course[self._chunk_idx]["learning goals"], "user_preferences": self._user_preferences}

    def post_run(self, data: any) -> ChatOutput:
        return ChatOutput(response=Message(role="ai", content=data.content), metadata={"metadata": "metadata"})
