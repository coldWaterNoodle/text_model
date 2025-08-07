# schemas/input_schema.py

from pydantic import BaseModel
from typing import Optional, Dict

class InputRequest(BaseModel):
    case_num: Optional[str] = "1"
    input_data: Optional[Dict] = None
