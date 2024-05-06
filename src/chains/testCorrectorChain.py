from langchain.chains.llm import LLMChain
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.testCorrector import TestCorrectorChainInput, TestCorrectorChainOutput

PROMPT = """
As an AI, you are tasked with evaluating a student's response to a question. You will be provided with the question, the student's answer, and the expected answer. Your task is to assign a grade from 0 to 100 and provide reasoning for the grade you assign. The output should be formatted as follows:

- Grade: A numerical grade between 0 and 100
- Reasoning: A detailed explanation for the grade assigned

Here's the format for the input and output:

**Input:**
- Question: ```{question}```
- Student's Answer: ```{student_answer}```
- Expected Answer: ```{expected_answer}```

Please adhere to the following formatting guidelines:
```{format_instructions}```
""".strip()


class Correction(BaseModel):
    grade: int = Field(..., description="Number grade from 0 to 100, where 0 is the worst and 100 is the best.")
    reasoning: str = Field(...)


class TestCorrectorChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM(model_parameters={"temperature": 0.1})
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Test corrector Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        super().build()
        built_llm = self._llm.build()

        output_parser = PydanticOutputParser(pydantic_object=Correction)
        output_parser = OutputFixingParser.from_llm(llm=built_llm, parser=output_parser)

        prompt = PromptTemplate(template=self.prompt, input_types={"question": "str", "student_answer": "str", "expected_answer": "str", "format_instructions": "str"},
                                input_variables=["question", "student_answer", "expected_answer"], partial_variables={"format_instructions": output_parser.get_format_instructions()},
                                output_parser=output_parser)

        self._chain = LLMChain(prompt=prompt, llm=built_llm, output_parser=output_parser)

    def run(self, data: dict | TestCorrectorChainInput) -> TestCorrectorChainOutput:
        pre = self.pre_run(data)
        result = self._chain.invoke(pre)
        return self.post_run(result)

    def pre_run(self, data: dict | TestCorrectorChainInput) -> dict:
        if isinstance(data, dict):
            data = TestCorrectorChainInput(**data)

        return {"question": data.question, "student_answer": data.student_answer, "expected_answer": data.expected_answer}

    def post_run(self, data: dict) -> TestCorrectorChainOutput:
        print(data["text"])
        return TestCorrectorChainOutput(grade=data["text"].grade, reasoning=data["text"].reasoning)
