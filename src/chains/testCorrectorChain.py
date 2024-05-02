from langchain.chains.llm import LLMChain
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.testCorrector import TestCorrectorChainInput, TestCorrectorChainOutput

PROMPT = """
You're an AI tasked with grading a student's answer on a question. To do this, you're given the following information:
- The question 
- The student's answer
- The expected answer

You're tasked with grading the answer by following this format:
- Grade: Number grade from 0 to 100
- Comment: A comment on the student's answer, formatted as if you're talking directly to the student
- Suggestions: A list of suggestions to improve the student's answer

*Input*
*Question:*
```{question}```

*Student's answer:*
```{student_answer}```

*Expected answer:*
```{expected_answer}```

Abide to the following formatting guidelines:
{format_instructions}
""".strip()


class Correction(BaseModel):
    grade: int = Field(..., description="Number grade from 0 to 100, where 0 is the worst and 100 is the best.")
    comment: str = Field(..., description="A comment on the student's answer, formatted as if you're talking directly to the student.")
    suggestions: list[str] = Field(..., description="A list of suggestions to improve the student's answer.")


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
        result = self._chain.run(pre)
        return self.post_run(result)

    def pre_run(self, data: dict | TestCorrectorChainInput) -> dict:
        if isinstance(data, dict):
            data = TestCorrectorChainInput(**data)

        return {"question": data.question, "student_answer": data.student_answer, "expected_answer": data.expected_answer}

    def post_run(self, data: dict) -> TestCorrectorChainOutput:
        return TestCorrectorChainOutput(grade=data["grade"], comment=data["comment"], suggestions=data["suggestions"])
