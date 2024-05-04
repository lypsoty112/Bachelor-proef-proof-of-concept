from langchain_core.prompts import PromptTemplate

from src.chains.baseChain import BaseChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.contentGenerator import ContentGeneratorInput

PROMPT = """
You're an AI tasked with teaching a HR-course to employees. You're given a piece of the course's content and the learning style of the employee. \
With this information equipped, write a text that is tailored to the employee's learning style and covers the course content. The text should be engaging and informative. \
Your text has to be formatted in markdown. Make sure to ask the employee if they have any questions at the end of the text.
Remember:
- You HAVE to use markdown to format your text.
- The provided course content is part of a larger course.
- There's no need to introduce yourself or the course content. Jump right into the content.
**Course Content:**
```
{course_content}
```

**Employee learning style:**
```
{employee_needs}
```
""".strip()


class ContentGeneratorChain(BaseChain):
    def __init__(self):
        super().__init__()
        self._llm = OpenAILLM(model_parameters={"temperature": 0.1, "max_tokens": 2000})
        self.llm_name: str = self._llm.llm_name
        self.chain_name: str = "contentGenerator Chain"
        self.prompt = PROMPT

    def build(self) -> None:
        super().build()
        built_llm = self._llm.build()
        promptTemplate = PromptTemplate(template=self.prompt, input_types={"course_content": "str", "employee_needs": "dict", "previous_interaction": "list"},
                                        input_variables=["course_content", "employee_needs", "previous_interaction"])

        self._chain = promptTemplate | built_llm

    def run(self, data: ContentGeneratorInput | dict) -> str:
        pre = self.pre_run(data)
        result = self._chain.invoke(pre)
        return self.post_run(result)

    def pre_run(self, data: ContentGeneratorInput | dict) -> dict:
        # Convert the data to the correct format
        if isinstance(data, dict):
            if "previous_interaction" not in data:
                data["previous_interaction"] = []
            data = ContentGeneratorInput(**data)

        previous_interaction = data.previous_interaction
        # Sort the messages by timestamp, where the first message is the oldest
        previous_interaction = [f"{msg.content}" for msg in previous_interaction][-2:]
        previous_interaction = "\n\n\n".join(previous_interaction)

        return {"course_content": data.course_content, "employee_needs": data.employee_needs, "previous_interaction": previous_interaction}

    def post_run(self, data: dict) -> str:
        return str(data)
