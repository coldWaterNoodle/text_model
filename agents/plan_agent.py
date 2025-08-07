import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict

from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

# 🔧 환경 및 모델 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel(model_name="models/gemini-1.5-flash")

class PlanAgent:
    """
    PlanAgent generates blog plans using templates and selects the best candidate.
    - Test mode: 사용자 입력 X (input data : 최근 CLI log 읽어와 사용, test_log/cli) -> 2-way 평가(evaluation) 진행
    - CORT mode: 사용자 입력 O (input_data : 직접 CLI 입력) → N-way 평가(evaluation) 진행

    Returns:
    - result_dict: Parsed JSON from best_output 
    - candidates: List of all candidate JSON strings
    - evaluation: Dict containing 'selected' index and 'reason' details
    - input_data: The dict used for generation (위 내용 참고)
    """
    def __init__(
            self,
            gen_template_path: str = "test_prompt/plan_generation_prompt.txt",
            eval2_template_path: str = "test_prompt/plan_evaluation_prompt.txt",
            nway_template_path: str = "test_prompt/plan_nway_evaluation_prompt.txt",
            default_nway_rounds: int = 3
        ):
            self.gen_template = self._load_template(gen_template_path, "생성")
            self.eval2_template = self._load_template(eval2_template_path, "2-way 평가")
            self.nway_template = self._load_template(nway_template_path, "N-way 평가")
            self.default_nway_rounds = default_nway_rounds

    def _load_template(self, path: str, name: str) -> str:
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"{name} 프롬프트 파일을 찾을 수 없습니다: {file}")
        return file.read_text(encoding="utf-8")

    def generate(
        self,
        input_data: Optional[dict] = None,
        mode: str = "cli",
        rounds: Optional[int] = None
    ) -> Tuple[Dict, Dict[str, str], Dict, dict]:
        loaded_input = self._prepare_input(input_data, mode)
        if rounds is None:
            rounds = self.default_nway_rounds if input_data is not None else 2

        # 디버깅용 출력 (plan_agent.py만 돌릴 때, run_agents.py로 돌릴 때에는 주석 처리)
        # print(f"🔍 [DEBUG] mode={mode}, rounds={rounds}")
        # print(json.dumps(loaded_input, indent=2, ensure_ascii=False))

        prompt = self._format_prompt(self.gen_template, loaded_input)
        raw_candidates = self._generate_candidates(prompt, rounds)
        # map to labels
        candidates = { f"후보 {i+1}": raw_candidates[i] for i in range(len(raw_candidates)) }

        if rounds == 2:
            best_output, selected, reason = self.evaluate_candidates(raw_candidates)
        else:
            best_output, selected, reason = self.evaluate_candidates_nway(raw_candidates)

        result = self._parse_json(best_output, raw_candidates, selected)
        evaluation_info = {"selected": selected, "reason": reason}
        return result, candidates, evaluation_info, loaded_input

    def _prepare_input(self, input_data: Optional[dict], mode: str) -> dict:
        if input_data is not None:
            return input_data
        if mode == "cli":
            logs = sorted(Path("test_logs/cli").glob("*_input_log.json"))
            if not logs:
                raise FileNotFoundError("CLI 입력 로그를 찾을 수 없습니다.")
            with open(logs[-1], "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError("유효한 입력이 제공되지 않았습니다.")

    def _format_prompt(self, template: str, input_data: dict) -> str:
        try:
            return template.format(**input_data)
        except KeyError as e:
            raise ValueError(f"프롬프트에 필요한 키가 누락되었습니다: {e}")
        except Exception as e:
            raise ValueError(f"프롬프트 포맷 중 오류 발생: {e}")

    def _clean_text(self, text: str) -> str:
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

    def _generate_candidates(self, prompt: str, rounds: int) -> List[str]:
        candidates: List[str] = []
        for i in range(rounds):
            raw = model.generate_content(prompt).text.strip()
            output = self._clean_text(raw)
            if output:
                candidates.append(output)
            else:
                print(f"⚠️ 후보 {i+1} 빈 응답")
        if len(candidates) < 2:
            raise ValueError(f"후보가 부족합니다: {len(candidates)}")
        return candidates

    def evaluate_candidates(self, candidates: List[str]) -> Tuple[str, str, dict]:
        prompt = self.eval2_template.format(
            candidate_1=candidates[0],
            candidate_2=candidates[1]
        )
        raw = model.generate_content(prompt).text.strip()
        eval_text = self._clean_text(raw)
        eval_result = json.loads(eval_text)
        sel = eval_result.get("selected", "").strip()
        reason = eval_result.get("reason", {})
        return candidates[int(sel[-1]) - 1], sel, reason

    def evaluate_candidates_nway(self, candidates: List[str]) -> Tuple[str, str, dict]:
        # 1) 각 후보를 블록 스트링으로 합치기
        blocks = "\n".join(f"후보 {i+1}:\n{cand}" for i, cand in enumerate(candidates))
        
        # 2) 포맷용 딕셔너리 생성
        format_args = {
            'n': len(candidates),
            'candidates': blocks
        }
        # 3) candidate_1, candidate_2… 키 추가
        for idx, cand in enumerate(candidates, start=1):
            format_args[f'candidate_{idx}'] = cand
        
        # 4) 템플릿에 키워드 인자로 넘겨서 포맷
        prompt = self.nway_template.format(**format_args)
        
        # 5) 평가 요청
        raw = model.generate_content(prompt).text.strip()
        eval_text = self._clean_text(raw)
        eval_result = json.loads(eval_text)
        sel = eval_result.get("selected", "").strip()
        reason = eval_result.get("reason", {})
        idx = int(sel.replace("후보", "").strip()) - 1
        return candidates[idx], sel, reason

    def _parse_json(self, best_output: str, candidates: List[str], selected: str) -> dict:
        try:
            return json.loads(best_output)
        except json.JSONDecodeError:
            idx = int(selected.replace("후보", "").strip()) - 1
            return json.loads(candidates[idx])

    def save_log(
        self,
        input_data: dict,
        candidates: Dict[str, str],
        best_output: str,
        selected: str,
        reason: dict,
        mode: str = "cli"
    ) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"test_logs/{mode}")
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{now}_plan_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "input": input_data,
                "candidates": candidates,
                "selected": selected,
                "best_output": best_output,
                "reason": reason
            }, f, ensure_ascii=False, indent=2)

# ✅ CLI 테스트
if __name__ == "__main__":
    print("🔍 PlanAgent CLI 테스트 시작")
    agent = PlanAgent()
    result, candidates, evaluation_info, input_data = agent.generate(mode="cli")
    agent.save_log(
        input_data=input_data,
        candidates=candidates,
        best_output=json.dumps(result, ensure_ascii=False),
        selected=evaluation_info["selected"],
        reason=evaluation_info["reason"],
        mode="cli"
    )
    print("\n🗂️ 결과가 test_logs/cli 에 저장되었습니다.")
    output = {**result, **evaluation_info}
    print(json.dumps(output, ensure_ascii=False, indent=2))

