import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

# 📦 InputAgent 클래스
class InputAgent:
    def __init__(
        self,
        input_data: Optional[dict] = None,
        case_num: str = "1",
        test_data_path: str = "test_data/test_input_onlook.json"
    ):
        self.case_num = case_num
        self.test_data_path = Path(test_data_path)
        self.input_data = input_data

    def collect(self, mode: str = "use") -> dict:
        """
        mode='test' 일 때: test_data_path 에서 case_num 으로 로드
        mode!='test' 일 때: CLI로 직접 입력받음
        """
        # 1) 외부에서 직접 data를 주입한 경우
        if self.input_data:
            return self.input_data

        # 2) TEST 모드: JSON 파일에서 불러오기
        if mode == "test":
            if not self.test_data_path.exists():
                raise FileNotFoundError(
                    f"테스트 입력 파일을 찾을 수 없습니다: {self.test_data_path}"
                )
            with open(self.test_data_path, encoding="utf-8") as f:
                data = json.load(f)
            case_key = f"test_case_{self.case_num}"
            if case_key not in data:
                raise ValueError(
                    f"{case_key} 항목을 {self.test_data_path}에서 찾을 수 없습니다."
                )
            return data[case_key]

        # 3) USE 모드: CLI 입력 (한 번만)
        return {
            "category":              input("카테고리를 입력하세요 (추후 선택으로 변경): "),
            "question1_concept":     input("Q1. 질환에 대한 개념 설명에서 강조되어야 할 메시지가 있나요?: "),
            "question2_condition":   input("Q2. 환자는 처음 내원 시 어떤 상태였나요?/증상입력: "),
            "question3_visit_photo": input("Q3. 내원 시 찍은 사진 업로드(파일명): "),
            "question4_treatment":   input("Q4. 치료 내용을 입력해주세요.: "),
            "question5_therapy_photo": input("Q5. 치료 과정 사진 업로드(콤마 구분): "),
            "question6_result":      input("Q6. 치료 결과에 대해 강조되어야 할 메시지가 있나요?: "),
            "question7_result_photo": input("Q7. 치료 결과 사진 업로드(파일명): "),
            "question8_extra":       input("Q8. 추가 강조 사항(환자 당부사항, 병원 철학 등): "),
        }

    def save_log(self, result: dict, mode: str = "use") -> None:
        """
        mode='test' -> test/logs 폴더에 저장
        mode='use'  -> use/logs 폴더에 저장
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"{mode}/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{now}_input_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


# ─── CLI 단독 실행 지원 ──────────────────────────────────────
if __name__ == "__main__":
    # standalone 실행 시에는 interactive로 모드를 묻습니다.
    mode = input("모드를 선택하세요 ('test' 또는 'use', 기본 'use'): ").strip().lower() or "use"
    if mode not in ("test", "use"):
        print("잘못된 모드 입력입니다. 'use'로 처리합니다.")
        mode = "use"

    case_num = "1"
    if mode == "test":
        case_num = input("테스트 케이스 번호를 입력하세요 (기본: 1): ").strip() or "1"

    agent = InputAgent(case_num=case_num)
    data = agent.collect(mode=mode)
    agent.save_log(data, mode=mode)
    print(json.dumps(data, ensure_ascii=False, indent=2))
