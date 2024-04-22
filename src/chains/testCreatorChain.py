from typing import Any, List

from langchain.output_parsers import OutputFixingParser, RetryOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM, OutputFixerOpenAILlm
from src.models.chains.testCreator import TestCreatorInput, TestCreatorOutput, Question as OutputQuestion

PROMPT = """
Your task is to generate a set of questions and answers based on a given portion of a Human Resources (HR) course and its associated learning goals. \
These questions will be used to assess the students' understanding of the course material. Please adhere to the following guidelines:

- Generate between {min_questions} to {max_questions} questions.
- Not all learning goals need to be addressed, but ensure that the most critical ones are covered.
- Each question should be followed by its corresponding answer.
- The questions should be designed in a way that they effectively test the students' comprehension of the learning goals.

Here is the format you should follow:

**Course:**
````{course}```

**Learning Goals:**
````{learning_goals}```
Now, based on the provided course content and learning goals, please generate the questions and answers.
Follow the requested formatting guidelines.
{format_instructions}

**Questions and Answers:**
"""


class Question(BaseModel):
    question: str = Field(description="The question to be asked")
    answer: str = Field(description="The answer to the question")


class Questions(BaseModel):
    questions: List[Question] = Field(description="A list of questions and answers")


class TestCreatorChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM()
        self._output_llm = OutputFixerOpenAILlm()
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "Test Creator Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        built_llm = self._llm.build()
        output_llm = self._output_llm.build()

        parser = PydanticOutputParser(pydantic_object=Questions)
        parser = OutputFixingParser.from_llm(llm=output_llm, parser=parser, max_retries=5)
        format_instructions = parser.get_format_instructions()

        template = PromptTemplate(input_types={"course": "str", "learning_goals": "str", "format_instructions": "str", "min_questions": "int", "max_questions": "int"},
                                  input_variables=["course", "learning_goals"], output_parser=parser, template=self.prompt,
                                  partial_variables={"format_instructions": format_instructions, "min_questions": 1, "max_questions": 7}, )

        self._chain = template | built_llm | parser

    def run(self, data: TestCreatorInput) -> TestCreatorOutput:
        pre_data = self.pre_run(data)
        retries = 0
        response = {}
        while retries < 3:
            try:
                response = self._chain.invoke(input={"course": pre_data["course"], "learning_goals": pre_data["learning_goals"]})
                break
            except Exception as e:
                self._logger.error(e)
                retries += 1

        if retries == 3:
            raise Exception("Failed to generate questions after 3 retries.")
        post_data = self.post_run(response)
        return post_data

    def pre_run(self, data: TestCreatorInput) -> dict:
        return {"course": data.course, "learning_goals": data.learning_goals, }

    def post_run(self, data: dict | Any) -> TestCreatorOutput:
        questions = data.questions
        return TestCreatorOutput(questions=[OutputQuestion(question=question.question, answer=question.answer) for question in questions])
