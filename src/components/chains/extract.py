from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

RESPONSE_TEMPLATE = """As an AI language model, your task is to extract specific information from a provided text. 
The information to be extracted will be clearly stated. If the information is not present in the text, 'value' 
should be None, and you should specify what's needed in the 'request' field. If no additional 
information is needed, the 'request' field should be None. Here's the format:

**Instructions**: Extract the following information from the text: {info_to_extract}

**Text**:
<TEXT>
{text}
</TEXT>

Please provide the extracted information in the following format:
{format_instructions}
""".strip()


class ExtractResponse(BaseModel):
    value: str | None = Field(None,
                              description="The extracted information from the text. If no information was "
                                          "found, this field should be None.")
    request: str | None = Field(None,
                                description="Questions or prompts for more information if necessary. If no extra "
                                            "information is needed, this field should be None.")


def extract_chain() -> Chain:
    llm = ChatOpenAI(
        temperature=0.3,
    )
    output_parser = JsonOutputParser(
        pydantic_object=ExtractResponse
    )

    format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        template=RESPONSE_TEMPLATE,
        parser=output_parser,
        input_types={
            "info_to_extract": str,
            "text": str,
            "format_instructions": str
        },
        input_variables=["info_to_extract", "text"],
        partial_variables={
            "format_instructions": format_instructions
        },
        output_key="response"
    )

    return LLMChain(
        prompt=prompt,
        llm=llm,
        output_parser=output_parser
    )
