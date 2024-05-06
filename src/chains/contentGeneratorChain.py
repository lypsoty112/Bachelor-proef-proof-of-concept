from langchain_core.prompts import PromptTemplate

from src.chains.baseChain import BaseChain
from src.chains.simpleTextTaskChain import SimpleTextTaskChain
from src.components.llm.openai import OpenAILLM
from src.models.chains.contentGenerator import ContentGeneratorInput

PROMPT = """
You're a teacher tasked with providing a HR-course to employees. To do this, you're given a piece of the course called `course chunk`, which is part of the overall course. \
Provide a Markdown formatted text that explains this chunk to employees with the goal of learning the course content. \
Make sure to use Markdown syntax to make the text as readable as possible. \


**Course Chunk:**
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
        self._simpleTextTaskChain = SimpleTextTaskChain()

    def build(self) -> None:
        super().build()
        self._simpleTextTaskChain.build()
        built_llm = self._llm.build()
        promptTemplate = PromptTemplate(template=self.prompt, input_types={"course_content": "str", "employee_needs": "dict", "previous_interaction": "list"},
                                        input_variables=["course_content", "employee_needs", "previous_interaction"])

        self._chain = promptTemplate | built_llm

    def run(self, data: ContentGeneratorInput | dict) -> str:
        pre = self.pre_run(data)
        result = self._chain.invoke(pre)
        result = self._simpleTextTaskChain.run({
            "text": result, "task": "Edit this text: Use markdown to highlight key points, sentences, and phrases."
        }).output
        return self.post_run({"content": result})

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
        return data["content"]
