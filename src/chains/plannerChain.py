from typing import List

from langchain.chains.llm import LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from src.chains.baseChain import BaseChain
from src.components.fileProcessor.pdfProcessor import PdfProcessor
from src.components.llm.openai import OpenAILLM, OutputFixerOpenAILlm
from src.components.splitter.RecursiveTextSplitter import RecursiveCharacterTextSplitterComponent
from src.models.chains.planner import PlannerOutput, PlannerInput

PROMPT = """Given the following text, your task is to extract and formulate learning goals for a student. Each learning goal should be phrased in the format: `The student can ...`. You may derive 
multiple learning goals from the provided text. Make sure your learning goals are specific, measurable, achievable, relevant, and time-bound (SMART). Respond with a bullet-point list of learning goals.
Make sure to answer in English, regardless of the language of the input text.

Here's an example of how the learning goals should be phrased: 
- The student can ...
- The student can ...
- The student can ...


Now, please extract the learning goals from the following text: 
Input:
```
{text}
```

Output:```
""".strip()


class PlannerChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM()
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Planner Chain"
        self.prompt = PROMPT
        self.components = {"processor": PdfProcessor(), "splitter": RecursiveCharacterTextSplitterComponent()}

    def build(self) -> None:
        for component in self.components.values():
            component.build()

        built_llm = self._llm.build()

        prompt = PromptTemplate(template=self.prompt, llm=built_llm, input_types={"text": "str"}, input_variables=["text"], )

        self._chain = LLMChain(prompt=prompt, llm=built_llm, )

    def run(self, data: PlannerInput) -> PlannerOutput:
        pre_data = self.pre_run(data)
        plan = []
        self._logger.info(f"Processed data, {len(pre_data)} chunks. Running the chain...")
        for chunks in pre_data:
            response = self._chain.invoke(input={"text": chunks.page_content})
            plan.append(response["text"])

        post_data = self.post_run({"plan": plan, "chunks": pre_data, })
        return post_data

    def pre_run(self, data: PlannerInput) -> List[Document]:
        file = data.file

        processed = self.components["processor"].run(data=file)
        return self.components["splitter"].run(data=processed)

    def post_run(self, data: dict) -> PlannerOutput:
        return PlannerOutput(plan=data["plan"], chunks=[{"page_content": chunk.page_content, "metadata": chunk.metadata} for chunk in data["chunks"]], )
