# agents/plan_agent_simple.py
# -*- coding: utf-8 -*-
"""
단일 Plan 생성기 (Refine/평가 없이 1회 생성)
- 최신 input 로그를 test_logs/use, test_logs/test에서 찾아 plan 1개 생성
- 로그가 없으면(선택) InputAgent로 바로 수집 시도
- 프롬프트 파일 없으면 내부 기본 프롬프트 사용
"""

import os, sys, json, re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- Env ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")
genai.configure(api_key=API_KEY)

# (선택) InputAgent
try:
    sys.path.append(str(Path(".").resolve()))
    from agents.input_agent_old2 import InputAgent  # noqa
except Exception:
    InputAgent = None

SEARCH_DIRS = [Path("test_logs/use"), Path("test_logs/test")]
GLOB_PATTERNS = ["*_input_log.json", "*_input.json"]
PROMPT_FILE = Path("test_prompt/plan_candidate_generation_prompt.txt")
OUTPUT_DIR = Path("test_logs/simple_plan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Utils ----------------
def now_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def latest_file_by_mtime(dir_: Path, pattern: str) -> Optional[Path]:
    files = list(dir_.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[-1]

def find_latest_input() -> Optional[Tuple[Dict[str, Any], str]]:
    """두 폴더 모두에서 다양한 패턴으로 최신 input 로그 탐색."""
    best: Tuple[Optional[Path], float] = (None, -1.0)
    for d in SEARCH_DIRS:
        if not d.exists():
            continue
        for pat in GLOB_PATTERNS:
            p = latest_file_by_mtime(d, pat)
            if p:
                mt = p.stat().st_mtime
                if mt > best[1]:
                    best = (p, mt)
    if best[0] is None:
        return None
    with open(best[0], "r", encoding="utf-8") as f:
        return json.load(f), str(best[0])

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

def load_prompt_text() -> str:
    if PROMPT_FILE.exists():
        return PROMPT_FILE.read_text(encoding="utf-8")
    # 내부 기본 프롬프트(섹션 5개를 한 번에 완성)
    return (
        "당신은 치과 케이스 블로그/랜딩 기획 전문가입니다.\n"
        "아래의 입력(input_data)을 바탕으로, '정확히 5개 섹션(서론/진단/치료/결과/관리)'을 갖춘 계획(plan) JSON을 한 번에 생성하세요.\n"
        "일부 입력이 비어 있더라도 임상 일반 절차/논리적 추론에 근거하여 자연스럽고 사실적인 서술로 보완하세요. 과장/보장 금지.\n"
        "\n"
        "[섹션 구조]\n"
        "1) 왜 이 글이 중요할까요? → 핵심 메시지 소개\n"
        "2) 진단/검사 포인트 → 내원 당시 상태와 검사 소견\n"
        "3) 치료는 이렇게 진행했어요 → 목표/재료/단계/내원 횟수(범위)\n"
        "4) 치료 결과와 회복 → 결과(합리적), 경과, 주의, 부작용 가능성\n"
        "5) 관리/예방 가이드 → 일상 관리와 재발 예방 팁\n"
        "\n"
        "[이미지] visit_images:0 / therapy_images:0 / result_images:0를 적절히 배치(없으면 생략)\n"
        "[링크정책] 본문에 지도 URL 금지(footer/meta만), 홈페이지는 본문 1회 언급 가능\n"
        "\n"
        "출력은 오직 하나의 JSON만:\n"
        "{\n"
        '  "title": string,\n'
        '  "summary": string,\n'
        '  "target_audience": string,\n'
        '  "persona_focus": string,\n'
        '  "sections": [ {5개 섹션, 각 섹션에 title/focus/description(3문장+)/key_points/tone/images/keywords/compliance_notes/evidence_hooks} ],\n'
        '  "keywords": [5~12개의 키워드],\n'
        '  "call_to_action": string,\n'
        '  "geo_branding": {"clinic_alias":"", "region_line":""},\n'
        '  "meta_panel": {"address":"", "phone":"", "homepage":"", "map_link":"", "treatment_period":""},\n'
        '  "link_policy": {"homepage_in_body_once": true, "map_in_footer_only": true}\n'
        "}\n"
        "\n"
        "[입력: input_data]\n"
        "{input_data_json}\n"
    )

def build_prompt(input_data: Dict[str, Any]) -> str:
    return load_prompt_text().replace(
        "{input_data_json}", json.dumps(input_data, ensure_ascii=False, indent=2)
    )

def save_plan(plan: Dict[str, Any], source_name: str) -> Path:
    ts = now_str()
    out_path = OUTPUT_DIR / f"{ts}_plan.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"source_input": source_name, "generated_at": ts, "plan": plan},
                  f, ensure_ascii=False, indent=2)
    return out_path

# ---------------- Main ----------------
def main():
    print("📌 Simple Plan 생성 시작")

    # 1) 최신 input 찾기
    found = find_latest_input()
    if found is None:
        print("⚠️ 입력 로그 파일이 없습니다. (검색 폴더: test_logs/use, test_logs/test)")
        if InputAgent is not None:
            yn = input("지금 InputAgent로 입력을 수집할까요? (Y/N): ").strip().lower()
            if yn == "y":
                ia = InputAgent()
                # 사용자가 원하는 쪽으로 고르세요: run_use() 또는 run_test()
                try:
                    data = ia.run_use()
                    src = "(via InputAgent.run_use())"
                except Exception:
                    data = ia.run_test()
                    src = "(via InputAgent.run_test())"
                input_data, source_name = data, src
            else:
                print("중단합니다."); return
        else:
            print("InputAgent가 없어 자동 수집이 불가합니다. 로그를 먼저 생성해 주세요.")
            return
    else:
        input_data, source_name = found

    # 2) 프롬프트 구성
    prompt = build_prompt(input_data)

    # 3) 모델 호출
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.6, max_output_tokens=8192, candidate_count=1, top_p=0.95, top_k=40
        )
    )

    # 4) JSON 파싱
    try:
        plan = extract_json(resp.text)
    except Exception:
        print("⚠️ JSON 파싱 실패. 모델 원문을 그대로 출력합니다:\n")
        print(resp.text)
        return

    # 5) 저장
    out_path = save_plan(plan, source_name)
    print(f"✅ Plan 저장 완료: {out_path}")

if __name__ == "__main__":
    main()
