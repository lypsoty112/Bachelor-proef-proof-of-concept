from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI, OpenAI

from src.components.llm.baseLLM import BaseLLM
from src.models.llm import LlmOutput


class OpenAILLM(BaseLLM):
    def __init__(self, chat: bool = False, model_parameters: dict = None) -> None:
        super().__init__(chat, model_parameters)
        self.llm_name = "openai"

    def build(self) -> BaseLanguageModel:
        return OpenAI(**self.model_parameters) if not self.chat else ChatOpenAI(**self.model_parameters)

    def post_run(self, data: object) -> LlmOutput:
        return LlmOutput(
            completion=str(data),
        )
