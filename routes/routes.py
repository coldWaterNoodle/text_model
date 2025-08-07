# routes/routes.py

from fastapi import APIRouter
from routes.input_route import router as input_router
from routes.plan_route import router as plan_router
from routes.title_route import router as title_router

# 🔹 개별 라우터 모으기
generate_sub_router = APIRouter()
generate_sub_router.include_router(input_router)
generate_sub_router.include_router(plan_router)
generate_sub_router.include_router(title_router)

# 🔹 /generate로 묶어서 외부에 export
router = APIRouter()
router.include_router(generate_sub_router, prefix="/generate")
