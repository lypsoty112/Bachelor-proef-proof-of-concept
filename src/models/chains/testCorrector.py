from pydantic import Field, BaseModel


class TestCorrectorChainInput(BaseModel):
    question: str = Field(..., description="The question")
    student_answer: str = Field(..., description="The student's answer")
    expected_answer: str = Field(..., description="The expected answer")


class TestCorrectorChainOutput(BaseModel):
    grade: int = Field(..., description="Number grade from 0 to 100, where 0 is the worst and 100 is the best.")
    comment: str = Field(..., description="A comment on the student's answer, formatted as if you're talking directly to the student.")
    suggestions: list[str] = Field(..., description="A list of suggestions to improve the student's answer.")
