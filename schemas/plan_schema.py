# schemas/plan_schema.py

from pydantic import BaseModel
from typing import List, Optional

class PlanInput(BaseModel):
    category: str
    question1_concept: str
    question2_condition: str
    question3_visit_photo: Optional[str] = None
    question4_treatment: str
    question5_therapy_photo: Optional[str] = None
    question6_result: str
    question7_result_photo: Optional[str] = None
    question8_extra: Optional[str] = None

class Section(BaseModel):
    title: str
    summary: str

class PlanOutput(BaseModel):
    title_guidance: str
    sections: List[Section]
