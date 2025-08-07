# routes/plan_route.py

# from fastapi import APIRouter
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import Optional, Dict
# from agents.plan_agent import PlanAgent
# from schemas.plan_schema import PlanInput

# router = APIRouter()

# class PlanRequest(BaseModel):
#     input_data: Optional[Dict] = None

# @router.post("/plan")  # ✅ "/generate/plan"이 아닌 "/plan"으로 수정
# async def generate_plan(body: PlanRequest):
#     try:
#         plan_agent = PlanAgent()
#         result = plan_agent.generate(body.input_data, mode="fastapi")
#         plan_agent.save_log(result, mode="fastapi")
#         return JSONResponse(content=result)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})


from fastapi import APIRouter
from fastapi.responses import JSONResponse
from agents.plan_agent import PlanAgent
from schemas.plan_schema import PlanInput

router = APIRouter()

# @router.post("/plan")
# async def generate_plan(input_data: PlanInput):
#     try:
#         plan_agent = PlanAgent()
#         result, candidates, best_output, best_index, parsed_input = plan_agent.generate(input_data.dict(), mode="fastapi")
#         plan_agent.save_log(parsed_input, candidates, best_output, best_index, mode="fastapi")
#         return JSONResponse(content=result)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"error": str(e)})

@router.post("/plan")
async def generate_plan(input_data: PlanInput):
    try:
        plan_agent = PlanAgent()
        result, candidates, best_output, best_index, parsed_input = plan_agent.generate(
            input_data.dict(exclude_none=True), mode="fastapi", rounds=2
        )
        plan_agent.save_log(parsed_input, candidates, best_output, best_index, mode="fastapi")
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})