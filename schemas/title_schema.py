# schemas/title_schema.py

from pydantic import BaseModel
from typing import List

class TitleRequest(BaseModel):
    sections: List[dict]  # each dict should contain 'title' and 'summary'

class TitleResponse(BaseModel):
    title: str
    candidates: List[str]
