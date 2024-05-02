from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.chat import ChatInput, ChatOutput

PROMPT = """
You're a chatbot tasked with answering questions from users about a HR-course. To help you, you're given the part of the course that covers the topic the user is asking about. \
You can use this information to answer the user's question, ask for more information, or ask the user to rephrase their question. \
If the answer is not in the course material, you can say that you don't know the answer. 


Make sure to always end your response by asking the user if they have any more questions, or if they'd rather continue the course.

**Course material:**
```
{course_material}
```
"""


class ChatChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM(True, model_parameters={"temperature": 0.2})
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Classifier Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        built_llm = self._llm.build()

        promptTemplate = ChatPromptTemplate.from_messages([SystemMessage(content=self.prompt), MessagesPlaceholder(variable_name="chat_history"), ])

        self._chain = promptTemplate | built_llm

    def run(self, data: ChatInput | dict) -> ChatOutput:
        pre = self.pre_run(data)
        result = self._chain.run(pre)
        return self.post_run(result)

    def pre_run(self, data: ChatInput | dict) -> dict:
        # Convert the data to the correct format
        if isinstance(data, dict):
            data = ChatInput(**data)

        course_material = data.metadata.get("course_material", None)
        if course_material is None:
            raise ValueError("Course material is missing from the metadata.")

        # Sort the messages by timestamp, where the first message is the oldest
        messages = sorted(data.messages, key=lambda x: x.timestamp)

        return {"course_material": course_material, "chat_history": messages[-10:]}

    def post_run(self, data: dict) -> ChatOutput:
        return ChatOutput(response=data["response"], metadata=data["metadata"])
