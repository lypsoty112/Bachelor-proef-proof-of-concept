from typing import List

from pydantic import BaseModel, Field


class Question(BaseModel):
    question: str = Field(..., description="The question to be asked")
    answer: str = Field(..., description="The answer to the question")


class TestCreatorInput(BaseModel):
    course: str = Field(..., description="The course content.")
    learning_goals: str = Field(..., description="The learning goals of the course.")


class TestCreatorOutput(BaseModel):
    questions: List[Question] = Field(..., description="The questions and answers generated from the course content and learning goals.")
