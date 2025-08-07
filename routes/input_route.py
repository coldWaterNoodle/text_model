# routes/input_route.py

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
from agents.input_agent import InputAgent

router = APIRouter()

class InputRequest(BaseModel):
    case_num: Optional[str] = "1"
    input_data: Optional[Dict] = None

@router.post("/input") # root 경로
async def generate_input(body: InputRequest):
    try:
        agent = InputAgent(input_data=body.input_data, case_num=body.case_num)
        result = agent.collect()
        agent.save_log(result, mode="fastapi")
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
