# agents/input_agent.py

import json
from pathlib import Path
from typing import Optional

# FastAPI 연동용
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class InputAgent:
    def __init__(self, input_data: Optional[dict] = None, case_num: str = "1", test_data_path: str = "data/test_input_data.json"):
        """
        - input_data: FastAPI 등 외부 입력을 통한 수집 (실사용 목적)
        - test_data_path: 테스트 입력용 JSON 파일
        - case_num: test_case_1, test_case_2 등 테스트 키 식별
        """
        self.case_num = case_num
        self.test_data_path = Path(test_data_path)
        self.input_data = input_data

    def collect(self) -> dict:
        if self.input_data:
            # 실 데이터 입력
            return self.input_data

        # 테스트 데이터 입력
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"{self.test_data_path} 파일이 존재하지 않습니다.")

        with open(self.test_data_path, encoding="utf-8") as f:
            data = json.load(f)

        case_key = f"test_case_{self.case_num}"
        if case_key not in data:
            raise ValueError(f"{case_key} 항목을 {self.test_data_path}에서 찾을 수 없습니다.")

        return data[case_key]


# ✅ CLI 테스트 용도
# if __name__ == "__main__":
#     print("🔍 InputAgent 테스트 시작")
#     agent = InputAgent(case_num="1", test_data_path="data/test_input_data.json")
#     result = agent.collect()

#     print(json.dumps(result, ensure_ascii=False, indent=2))

# ✅ FastAPI 연동
app = FastAPI()

@app.post("/generate/input")
async def generate_input(request: Request):
    """
    POST /generate/input
    body: {
        "case_num": "1"   (optional),
        "input_data": {...}  (optional)
    }
    """
    try:
        body = await request.json()
        input_data = body.get("input_data")
        case_num = body.get("case_num", "1")

        agent = InputAgent(input_data=input_data, case_num=case_num)
        result = agent.collect()
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
