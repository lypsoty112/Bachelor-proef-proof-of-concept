from typing import Dict

from pydantic import BaseModel, Field


class ContentGeneratorInput(BaseModel):
    previous_interaction: list = Field(..., description="The previous interaction between the AI and the employee.")
    course_content: str = Field(..., description="The generated content for the employees based on the course content and employee needs.")
    employee_needs: Dict[str, str] = Field(..., description="The needs of the employees for the course content.")
