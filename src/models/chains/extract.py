from pydantic import BaseModel, Field


class ExtractOutput(BaseModel):
    value: str | None = Field(None, description="The extracted information from the text. If no information was "
                                                "found, this field should be None.")
    request: str | None = Field(None, description="Questions or prompts for more information if necessary. If no extra "
                                                  "information is needed, this field should be None.")


class ExtractInput(BaseModel):
    info_to_extract: str = Field(..., description="The information to be extracted from the text.")
    text: str = Field(..., description="The text from which the information should be extracted.")

