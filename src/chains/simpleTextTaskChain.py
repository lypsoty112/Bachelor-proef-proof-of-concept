from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.simpleTextTask import SimpleTextTaskInput, SimpleTextTaskOutput
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain

PROMPT = """
I'm going to provide you with a piece of text and a specific task. Your job is to perform the task on the given text, following the instructions as accurately as possible. Please ensure your response adheres to the requirements of the task.

Here's an example to illustrate:

Input Text: "The quick brown fox jumps over the lazy dog."
Task: "Identify the animals in the text."
"fox, dog"

Now, let's proceed with the actual task:

Input Text: "{text}"
Task: "{task}"
"""


class SimpleTextTaskChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM()
        self.llm_name = self._llm.llm_name
        self.chain_name = "simple text task"
        self.prompt = PROMPT
        self._chain = None

    def build(self) -> None:
        built_llm = self._llm.build()

        prompt = PromptTemplate(template=PROMPT, input_types={"text": str, "task": str, }, input_variables=["text", "task"], output_key="text", )

        self._chain = LLMChain(llm=built_llm, prompt=prompt, )

    def pre_run(self, data: SimpleTextTaskInput | dict) -> dict:
        if isinstance(data, dict):
            data = SimpleTextTaskInput(**data)
        return {"text": data.text, "task": data.task, }

    def run(self, data: SimpleTextTaskInput | dict) -> SimpleTextTaskOutput:
        pre_data = self.pre_run(data)
        response = self._chain.invoke(pre_data)
        return self.post_run(response)

    def post_run(self, data: dict) -> SimpleTextTaskOutput:
        return SimpleTextTaskOutput(output=data["text"], )
