from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_core.pydantic_v1 import BaseModel, Field


PROMPT = """
Your task is to engage in a conversation with the user and gather specific information. This information will be used to populate a JSON schema. If the user does not provide the necessary information, you should prompt them again. If they still do not provide the information, the corresponding value in the JSON schema should be set to None. Remember, you must always pose a question to the user to gather information. 

Here's the structure of the task:

**Conversation:**
{conversation}

**JSON Object:**
{json_object}

**Output Schema:**
{format_instructions}

For example, if the conversation is about gathering user's name and age, and the user doesn't provide age, the output schema should look like this:

**Conversation:**
- AI: What's your name?
- User: My name is John.
- AI: How old are you?
- User: I'd rather not say.

**JSON Object:**
{{
  "name": null,
  "age": null
}}

**Output:**
{{
  "question": "You didn't provide your age. Could you please provide it?",
  "updated_object": {{
    "name": "John",
    "age": null
  }}
}}
Please follow this pattern to complete the task.
"""


def question_extract_chain(object_to_extract: Any) -> Chain:
    llm = ChatOpenAI(
        temperature=0.5,
    )

    class ExpectedOutput(BaseModel):
        question: str = Field(None, description="The question to ask the user.")
        updated_object: object_to_extract

    output_parser = JsonOutputParser(
        pydantic_object=ExpectedOutput
    )
    format_instructions = output_parser.get_format_instructions()

    output_parser = OutputFixingParser.from_llm(
        llm=llm,
        parser=output_parser
    )

    prompt = PromptTemplate(
        template=PROMPT,
        parser=output_parser,
        input_types={
            "conversation": str,
            "json_object": str,
            "format_instructions": str
        },
        input_variables=["json_object", "conversation"],
        partial_variables={
            "format_instructions": format_instructions
        },
        output_key="text"
    )

    return LLMChain(
        prompt=prompt,
        llm=llm,
        output_parser=output_parser
    )
