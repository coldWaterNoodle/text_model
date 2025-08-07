# routes/title_route.py

from fastapi import APIRouter, HTTPException
from agents.title_agent import TitleAgent
from schemas.title_schema import TitleRequest, TitleResponse

router = APIRouter()

@router.post("/generate/title", response_model=TitleResponse)
def generate_title(input_data: TitleRequest):
    try:
        agent = TitleAgent()
        result, candidates, best_output = agent.generate(input_data=input_data.dict(), use_previous_log=False, mode="fastapi")
        agent.save_log(result, candidates, best_output, mode="fastapi")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
