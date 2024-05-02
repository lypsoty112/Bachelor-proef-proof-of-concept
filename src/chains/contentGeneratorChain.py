import random

from langchain_core.prompts import PromptTemplate

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.contentGenerator import ContentGeneratorInput

PROMPT = """
As an AI content generator, your task is to generate content for HR-courses aimed at employees. The content should be easily accessible, understandable, and tailored to the specific needs of \
the employees. The content should be formatted and structured using Markdown. 

Here's how you should structure your response:

1. Start with a brief introduction about the course content.
2. Break down the course content into sections or subtopics.
3. For each section, provide a clear and concise explanation, making sure to address the employees' needs.
4. Use bullet points, headings, and subheadings to organize the content.
5. End with a summary or key takeaways from the course content.

**Course Content:**
```
{course_content}
```
**Employee Needs:**
```
{employee_needs}
```
""".strip()


class ContentGeneratorChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM(model_parameters={"temperature": 0.3, "max_tokens": 2000})
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "contentGenerator Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        super().build()
        built_llm = self._llm.build()
        promptTemplate = PromptTemplate(template=self.prompt, input_types={"course_content": "str", "employee_needs": "dict"}, input_variables=["course_content", "employee_needs"])

        self._chain = promptTemplate | built_llm

    def run(self, data: ContentGeneratorInput | dict) -> str:
        pre = self.pre_run(data)
        result = self._chain.invoke(pre)
        return self.post_run(result)

    def pre_run(self, data: ContentGeneratorInput | dict) -> dict:
        # Convert the data to the correct format
        if isinstance(data, dict):
            data = ContentGeneratorInput(**data)

        return {"course_content": data.course_content, "employee_needs": data.employee_needs}

    def post_run(self, data: dict) -> str:
        return str(data)
