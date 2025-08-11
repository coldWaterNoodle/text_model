# agents/title_agent.py
# -*- coding: utf-8 -*-
"""
TitleAgent (plan 결과 기반 제목 생성)
- 모드:
  1) test: test_logs/test 최신 *_plan_log.json 기반 생성
     - 없으면 최신 input으로 plan 생성 후 진행(사용자 동의)
  2) use : ① 기존 선택(로그 기반) ② 직접 입력(새 수집) → plan 생성 후 진행
- 평가 에이전트 없음: 모델이 '후보 생성 + 최종 선택'까지 수행
- 프롬프트 파일이 없으면 안전한 기본 프롬프트 사용
- 저장: *_title_log.json(전체 후보·선택결과), *_title.json(요약 메타)
"""

import os, sys, re, json, random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# deps
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# 프로젝트 루트 import 경로
sys.path.append(str(Path(".").resolve()))

# plan 생성 및 입력 수집을 위해 활용(있으면 사용, 없으면 안내)
try:
    from agents.plan_agent import (
        PlanAgent as _PlanAgent,  # plan 생성 파이프라인 재사용
        validate_plan as _validate_plan,
    )
    _HAS_PLAN_AGENT = True
except Exception:
    _PlanAgent = None
    _validate_plan = None
    _HAS_PLAN_AGENT = False

try:
    from agents.input_agent import InputAgent as _InputAgent
    _HAS_INPUT_AGENT = True
except Exception:
    _InputAgent = None
    _HAS_INPUT_AGENT = False


# -------------- utils --------------
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def read_json(p: Path) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def latest_file_by_mtime(dir_: Path, pattern: str) -> Optional[Path]:
    files = list(dir_.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]

def latest_plan_log(dir_: Path) -> Optional[Path]:
    return latest_file_by_mtime(dir_, "*_plan_log.json")


# -------------- JSON extractor --------------
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("빈 모델 응답")
    m = JSON_BLOCK_RE.search(text)
    if m:
        return json.loads(m.group(1))
    start = text.find("{")
    if start == -1:
        raise ValueError("JSON 시작 '{' 없음")
    stack = 0; end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{": stack += 1
        elif ch == "}":
            stack -= 1
            if stack == 0:
                end = i + 1
                break
    if end == -1:
        raise ValueError("JSON 중괄호 불일치")
    return json.loads(text[start:end])


# -------------- Gemini --------------
class GeminiClient:
    def __init__(self, model="gemini-1.5-pro", temperature=0.6, max_output_tokens=2048):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")
        genai.configure(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        m = genai.GenerativeModel(self.model)
        resp = m.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature if temperature is None else temperature,
                max_output_tokens=self.max_output_tokens,
                candidate_count=1,
                top_p=0.95, top_k=40,
            )
        )
        if getattr(resp, "text", None):
            return resp.text
        if getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            if parts and getattr(parts[0], "text", ""):
                return parts[0].text
        raise ValueError("응답에 text 없음")


# -------------- Prompt pack --------------
class PromptPack:
    def __init__(self):
        pref = Path("test_prompt")
        if not pref.exists():
            pref = Path("prompts")
        self.dir = pref
        self.title_prompt_path = pref / "title_generation_prompt.txt"

    def template(self) -> str:
        if self.title_prompt_path.exists():
            return read_text(self.title_prompt_path)
        raise FileNotFoundError(f"{self.title_prompt_path} 프롬프트 파일이 없습니다.")

    def fill(self, tmpl: str, plan: Dict[str, Any], n: int) -> str:
        safe_plan = json.dumps(plan, ensure_ascii=False)
        return tmpl + f"\n\nN={n}\nplan={safe_plan}\n"



# -------------- TitleAgent --------------
class TitleAgent:
    def __init__(self, model="gemini-1.5-pro"):
        self.prompts = PromptPack()
        self.gemini = GeminiClient(model=model)

    def generate(self, plan: Dict[str, Any], n_candidates: int = 8) -> Dict[str, Any]:
        tmpl = self.prompts.template()
        prompt = self.prompts.fill(tmpl, plan, n_candidates)
        raw = self.gemini.generate_text(prompt, temperature=0.65)
        obj = extract_json(raw)

        # 필수 필드 보정
        obj.setdefault("candidates", [])
        obj.setdefault("selected", {})
        if not obj["selected"].get("title") and obj["candidates"]:
            # 모델이 선택을 못했을 때 폴백: 첫 후보 선택
            obj["selected"] = {
                "title": obj["candidates"][0].get("title", "").strip(),
                "why_best": "모델 미선정 → 1순위 후보 폴백"
            }
        return obj


# -------------- Save helpers --------------
def save_title(base_dir: Path, plan_path: str, mode_label: str, result: Dict[str, Any]) -> None:
    ts = now_str()
    # 전체 후보/선정결과 → *_title_log.json
    write_json(base_dir / f"{ts}_title_log.json", result)
    # 메타(가벼운 요약) → *_title.json
    meta = {
        "mode": mode_label,
        "plan_path": plan_path,
        "ts": ts,
        "selected_title": (result.get("selected", {}) or {}).get("title", "")
    }
    write_json(base_dir / f"{ts}_title.json", meta)
    print(f"✅ 저장 완료: {base_dir/(ts+'_title_log.json')} (candidates+selected)")
    print(f"✅ 저장 완료: {base_dir/(ts+'_title.json')} (meta)")


# -------------- Orchestrations --------------
def _ensure_plan_from_latest_input(base_dir: Path, which_mode_label: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    최신 plan이 없을 때, 사용자의 동의가 있으면 InputAgent + PlanAgent로 즉시 생성.
    반환: (plan_dict, plan_path_str)
    """
    if not _HAS_PLAN_AGENT or not _HAS_INPUT_AGENT:
        print("⚠️ plan 또는 input 모듈을 찾을 수 없습니다. 먼저 plan_agent를 실행해 plan을 만들어주세요.")
        return None, None

    yn = input("최신 plan이 없습니다. 지금 생성할까요? (Y/N): ").strip().lower()
    if yn != "y":
        return None, None

    ia = _InputAgent()
    if which_mode_label == "test":
        data = ia.run_test()
        base = Path("test_logs/test")
        mode_label = "test-generate"
        round_label = "round2"
    else:
        data = ia.run_use()
        base = Path("test_logs/use")
        mode_label = "use-generate"
        round_label = "auto(3-5)"

    # 간결한 plan 생성 파이프라인(TitleAgent 내 인라인 구성)
    pa = _PlanAgent()
    cands = pa.generate_candidates(data, k=4)
    if not cands:
        print("⚠️ plan 후보 생성 실패"); return None, None
    # 간단 점수(로컬)로 베스트 추정
    def _local_adherence(plan: Dict[str, Any]) -> int:
        secs = plan.get("sections", [])
        score = 100 if isinstance(secs, list) and len(secs) >= 5 else 80
        return score
    cands.sort(key=lambda c: _local_adherence(c.get("plan", {})), reverse=True)
    best = cands[0]["plan"]
    refined = pa.refine_to_five_sections(data, best, strict=False)
    ok, errs = _validate_plan(refined) if _validate_plan else (True, [])
    if not ok:
        refined = pa.refine_to_five_sections(data, refined, strict=True)
        ok, errs = _validate_plan(refined) if _validate_plan else (True, [])
    if not ok:
        print("⚠️ plan 최종 검증 실패:", errs); return None, None

    # plan 저장(TitleAgent에서는 경량 저장)
    ts = now_str()
    write_json(base / f"{ts}_plan_log.json", refined)
    write_json(base / f"{ts}_plan.json", {
        "mode": mode_label, "round": round_label, "input_path": "(inline via TitleAgent)", "ts": ts,
        "selected_id": "cand_inline", "selection_reason": {"score_0_100": 100, "reason_summary": "inline-best"}
    })
    print(f"ℹ️ TitleAgent가 plan을 생성했습니다 → {base/(ts+'_plan_log.json')}")
    return refined, str(base / f"{ts}_plan_log.json")


def run_test() -> Optional[Dict[str, Any]]:
    base_dir = Path("test_logs/test")
    plan_path = latest_plan_log(base_dir)
    if not plan_path:
        plan, plan_src = _ensure_plan_from_latest_input(base_dir, which_mode_label="test")
        if plan is None:
            print("생성을 중단합니다."); return None
        plan_path = Path(plan_src)
    else:
        plan = read_json(plan_path)

    agent = TitleAgent()
    result = agent.generate(plan, n_candidates=8)
    save_title(base_dir, str(plan_path), "test", result)
    return result


def run_use() -> Optional[Dict[str, Any]]:
    print("\nuse 데이터 소스를 선택하세요:")
    print("1) 기존 선택(로그 기반)  2) 직접 입력(새로 수집)")
    src = input("선택 (1/2): ").strip()

    base_dir = Path("test_logs/use")

    if src == "1":
        plan_path = latest_plan_log(base_dir)
        if not plan_path:
            plan, plan_src = _ensure_plan_from_latest_input(base_dir, which_mode_label="use")
            if plan is None:
                print("생성을 중단합니다."); return None
            plan_path = Path(plan_src)
        else:
            plan = read_json(plan_path)

    elif src == "2":
        plan, plan_src = _ensure_plan_from_latest_input(base_dir, which_mode_label="use")
        if plan is None:
            print("생성을 중단합니다."); return None
        plan_path = Path(plan_src)
    else:
        print("⚠️ 잘못된 입력"); return None

    agent = TitleAgent()
    result = agent.generate(plan, n_candidates=8)
    save_title(base_dir, str(plan_path), "use", result)
    return result


def run_latest_title_only(base_dir: Path) -> Optional[Dict[str, Any]]:
    latest = latest_file_by_mtime(base_dir, "*_title_log.json")
    if not latest:
        print(f"⚠️ 최신 title 결과가 없습니다: {base_dir}/*_title_log.json")
        meta = latest_file_by_mtime(base_dir, "*_title.json")
        if meta:
            print(f"ℹ️ 참고: 최신 메타 파일은 있습니다 → {meta.name}")
        return None
    obj = read_json(latest)
    sel = (obj.get("selected", {}) or {}).get("title", "")
    print(f"📄 최신 TITLE: {latest.name}")
    print("selected:", sel)
    return obj


def main():
    print("$ python agents/title_agent.py")
    print("📝 TitleAgent 시작\n")
    print("모드:")
    print("1) test        → 최신 plan 로그 기반 제목 생성(없으면 생성 안내)")
    print("2) use         → ① 기존 선택(로그) ② 직접 입력(새 수집) 후 생성")
    print("3) latest-view → 최신 title 로그 요약 보기(폴더 선택)")
    sel = input("선택 (1/2/3): ").strip()
    try:
        if sel == "1":
            run_test()
        elif sel == "2":
            run_use()
        elif sel == "3":
            which = input("폴더 선택 (1: test_logs/test, 2: test_logs/use): ").strip()
            base = Path("test_logs/test") if which == "1" else Path("test_logs/use")
            run_latest_title_only(base)
        else:
            print("⚠️ 잘못된 입력"); sys.exit(1)
    except Exception as e:
        print("❌ 오류:", e); sys.exit(1)

if __name__ == "__main__":
    main()
