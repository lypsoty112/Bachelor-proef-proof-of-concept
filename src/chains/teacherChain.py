from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from src.chains.baseChain import BaseChain
from src.components.llm.baseLLM import BaseLLM
from src.components.llm.openai import OpenAILLM
from src.models.chains.chat import ChatInput, ChatOutput

PROMPT = """
As an AI language model simulating a teacher's role, you are tasked with delivering a piece of course material to a student. You have been provided with the specific content of the material, the \
general learning objectives associated with this material, and the student's preferred learning style. Your task is to adapt your teaching approach to align with the student's learning preferences. 
""".strip()

PROMPT2 = """
Course material: {course_material}

Learning goals: {learning_goals}

Student's learning preferences: {learning_preferences}
""".strip()


class TeacherChain(BaseChain):
    def __init__(self):
        super().__init__()
        self.chain_name: str = "Teacher chain (chat)"  # The name of the chain. Useful for debugging
        self.prompt: str = PROMPT  # The prompt that the chain uses to generate completions
        self._llm: BaseLLM | None = OpenAILLM(chat=True, )  # The LLM that the chain uses
        self.llm_name: str = self._llm.llm_name
        self._chain: Runnable | None = None  # The chain that the chain uses. This is created in the build method

    def build(self) -> None:
        built_llm = self._llm.build()

        prompt = ChatPromptTemplate.from_messages([("system", PROMPT), MessagesPlaceholder(variable_name="previous messages", ), ("system", PROMPT2), ("human", "{last message}")])

        self._chain = LLMChain(prompt=prompt, llm=built_llm)

    def run(self, data: ChatInput) -> ChatOutput:
        pre = self.pre_run(data)
        result = self._chain.invoke(pre)
        return self.post_run(result)

    def pre_run(self, data: ChatInput) -> dict:
        return {

        }

    def post_run(self, data: dict) -> ChatOutput:
        return {

        }
