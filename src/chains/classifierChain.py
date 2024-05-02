from enum import Enum

from langchain.chains.llm import LLMChain
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.classifier import ClassifierInput, NextStep, ClassifierOutput as ClassifierChainOutput

PROMPT = """
You're an AI assistant whose an expert in logical reasoning and language understanding. \
You're part of a system that attempts to teach a HR-course to employees. \
You're responsible for analysing the customer's message and providing the necessary information to the rest of the system in order for the system to make informed decisions. \

More specifically, you're responsible for determining the following information from the customer's message:
- Next step: The next step the system should take based on the customer's message.

**Here's the customer's message to analyse:**
```{message}```

**For context, here's the chat history between the customer and the system:**
```
{chat_history}
```

**Provide the information by following these instructions:**
{format_instructions}
""".strip()

NEXTSTEP_INFO = {
    NextStep.CONTINUE_COURSE: {
        "description": "The system should continue the course because the customer indicates they understand the course material, or ask to continue the course."
    }, NextStep.ASK_FOR_MORE_INFORMATION: {
        "description": "The system should ask the customer for more information. "
                       "This could be because the customer's message is unclear, or the system needs more information to make an informed decision."
    }, NextStep.RESPOND_TO_QUESTION: {
        "description": "The system should respond to the customer's question because the customer has asked a question. ",
    }

}


class ClassifierOutput(BaseModel):
    next_step: NextStep = Field(..., description=f"The next step the system should take based on the customer's message.")
    explanation: str = Field(..., description="An explanation of why the system should take the next step.")

class ClassifierChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM(model_parameters={"temperature": 0.1})
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Classifier Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        built_llm = self._llm.build()

        output_parser = PydanticOutputParser(pydantic_object=ClassifierOutput)
        output_parser = OutputFixingParser.from_llm(llm=built_llm, parser=output_parser)

        prompt = PromptTemplate(template=self.prompt, input_types={"message": "str", "chat_history": "list", "format_instructions": "str"}, input_variables=["message", "chat_history"],
                                output_parser=output_parser, partial_variables={"format_instructions": output_parser.get_format_instructions()})

        self._chain = LLMChain(prompt=prompt, llm=built_llm, output_parser=output_parser)

    def run(self, data: dict | ClassifierInput) -> ClassifierChainOutput:
        pre = self.pre_run(data)
        response = self._chain.invoke(pre)
        return self.post_run(response)

    def pre_run(self, data: dict | ClassifierInput) -> dict:
        # Initialize classifierInput
        if isinstance(data, dict):
            data = ClassifierInput(**data)

        return {"message": data.message, "chat_history": data.chat_history[-10:]}

    def post_run(self, data: dict) -> ClassifierChainOutput:
        return ClassifierChainOutput(next_step=data["text"].next_step)
