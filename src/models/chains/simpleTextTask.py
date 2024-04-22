from pydantic import BaseModel, Field


class SimpleTextTaskInput(BaseModel):
    text: str = Field(description="The text to be processed.")
    task: str = Field(description="The task to be performed on the text.")


class SimpleTextTaskOutput(BaseModel):
    output: str = Field(description="The output of the task.")
