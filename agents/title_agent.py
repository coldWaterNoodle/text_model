import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
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

class TitleAgent:
    """
    TitleAgent generates blog titles using external prompt templates and selects the best via 2-way or N-way evaluation.
    - Test mode: rounds=2 (2-way)
    - Use mode: rounds=default_nway_rounds (N-way)

    generate(input_data, mode, rounds) -> (best_title, candidates_dict, evaluation_info, input_data)
    """
    def __init__(
        self,
        gen_template_path: str = "test_prompt/title_generation_prompt.txt",
        eval2_template_path: str = "test_prompt/title_evaluation_prompt.txt",
        eval_nway_template_path: str = "test_prompt/title_nway_evaluation_prompt.txt",
        default_nway_rounds: int = 3
    ):
        # Load external prompt templates
        self.gen_template = self._load_template(gen_template_path, "제목 생성")
        self.eval2_template = self._load_template(eval2_template_path, "2-way 평가")
        self.eval_nway_template = self._load_template(eval_nway_template_path, "N-way 평가")
        self.default_nway_rounds = default_nway_rounds

    def _load_template(self, path: str, name: str) -> str:
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"{name} 프롬프트 파일을 찾을 수 없습니다: {path}")
        return file.read_text(encoding="utf-8")

    def generate(
        self,
        input_data: dict,
        mode: str = "use",
        rounds: Optional[int] = None
    ) -> Tuple[str, Dict[str,str], Dict, dict]:
        # Determine number of candidates
        if rounds is None:
            rounds = 2 if mode == "test" else self.default_nway_rounds

        # Generate multiple title candidates to avoid index errors
        gen_prompt = self.gen_template.format(**input_data)
        candidates = self._generate_candidates(gen_prompt, rounds)
        candidates_dict = {f"후보 {i+1}": candidates[i] for i in range(len(candidates))}

        # Evaluate candidates
        if rounds == 2:
            best, sel, reason = self._eval_2way(candidates)
        else:
            best, sel, reason = self._eval_nway(candidates)

        return best, candidates_dict, {"selected": sel, "reason": reason}, input_data

    def _generate_candidates(self, prompt: str, rounds: int) -> List[str]:
        """Generate unique candidates by multiple calls."""
        candidates: List[str] = []
        attempts = rounds * 3
        for _ in range(attempts):
            resp = model.generate_content(prompt)
            txt = resp.text.strip()
            # Each line could be separate candidate, but treat full output uniquely
            if txt and txt not in candidates:
                candidates.append(txt)
            if len(candidates) >= rounds:
                break
        if len(candidates) < rounds:
            raise ValueError(f"타이틀 후보가 충분히 생성되지 않았습니다: {len(candidates)}")
        return candidates[:rounds]

    def _eval_2way(self, candidates: List[str]) -> Tuple[str, str, str]:
        prompt = self.eval2_template.format(
            candidate_1=candidates[0],
            candidate_2=candidates[1]
        )
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        try:
            doc = json.loads(txt)
            sel = doc.get("selected", "").strip()
            reason = doc.get("reason", "").strip()
        except:
            m = re.search(r"후보\s*([12])", txt)
            sel = f"후보 {m.group(1)}" if m else "후보 1"
            reason = txt
        idx = int(sel.replace("후보", "")) - 1
        return candidates[idx], sel, reason

    def _eval_nway(self, candidates: List[str]) -> Tuple[str, str, str]:
        # Build block listing for all candidates
        blocks = "\n".join(f"후보 {i+1}: {c}" for i, c in enumerate(candidates))
        # Prepare format args including count and block
        fmt_args = {'n': len(candidates), 'candidates': blocks}
        # Add individual placeholders for safety
        for i, c in enumerate(candidates, start=1):
            fmt_args[f'candidate_{i}'] = c
        prompt = self.eval_nway_template.format(**fmt_args)
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        try:
            doc = json.loads(txt)
            sel = doc.get("selected", "").strip()
            reason = doc.get("reason", "").strip()
        except:
            m = re.search(r"후보\s*(\d+)", txt)
            num = m.group(1) if m else "1"
            sel = f"후보 {num}"
            reason = txt
        idx = int(sel.replace("후보", "")) - 1
        return candidates[idx], sel, reason

    def save_log(
        self,
        input_data: dict,
        candidates: Dict[str,str],
        best_output: str,
        selected: str,
        reason: str,
        mode: str = "cli"
    ) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = Path(f"test_logs/{mode}")
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / f"{now}_title_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "input": input_data,
                "candidates": candidates,
                "selected": selected,
                "reason": reason,
                "best_output": best_output
            }, f, ensure_ascii=False, indent=2)

# CLI 테스트
if __name__ == "__main__":
    from input_agent import InputAgent
    plan = InputAgent().collect(mode="cli")
    agent = TitleAgent()
    best, cands, info, _ = agent.generate(plan, mode="test")
    agent.save_log(plan, cands, best, info['selected'], info['reason'], mode="cli")
    print(f"🔍 결과: {best}")
