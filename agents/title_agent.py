# -*- coding: utf-8 -*-
"""
TitleAgent (카테고리/지역/페르소나 반영 · 의료광고 준수 · 모델 자체선정)
- 입력: PlanAgent 산출물(plan dict 또는 *_plan.json 경로/자동탐색)
- 프롬프트: test_prompt/title_generation_prompt.txt, test_prompt/title_evaluation_prompt.txt (없으면 내장 프롬프트 사용)
- 모델: Gemini (GEMINI_API_KEY 필요) — JSON 강인 파싱/리페어 포함
- 출력: test_logs/{mode}/{YYYYMMDD}/{timestamp}_title.json (최종) + {timestamp}_title_log.json (로그)

동작 개요
1) plan을 수집/정규화 → 제목 생성 프롬프트 구성
2) 후보 N개 생성(candidates)
3) 평가 프롬프트로 모델이 최적 1개 선택(selected)
4) 스키마/규칙 검증 후 저장
"""

from __future__ import annotations

import os, json, re, ast
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import time

# -----------------------
# 환경 & 모델
# -----------------------
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------
# 경로 유틸
# -----------------------
GEN_PROMPT_PATH = Path("test_prompt/title_generation_prompt.txt")
EVAL_PROMPT_PATH = Path("test_prompt/title_evaluation_prompt.txt")

DEF_MODE = "use"


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -----------------------
# 모델 클라이언트
# -----------------------
class GeminiClient:
    def __init__(self, model: str = "models/gemini-1.5-flash", temperature: float = 0.7, max_output_tokens: int = 2048):
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = 3
        self.retry_delay = 1.0

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        for attempt in range(self.max_retries):
            try:
                m = genai.GenerativeModel(self.model_name)
                resp = m.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature if temperature is None else temperature,
                        max_output_tokens=self.max_output_tokens,
                        candidate_count=1,
                        top_p=0.95,
                        top_k=40,
                    ),
                )
                # 일반 텍스트 경로
                if getattr(resp, "text", None):
                    return resp.text
                # 후보 파츠 경로(보수용)
                if getattr(resp, "candidates", None):
                    parts = getattr(resp.candidates[0].content, "parts", [])
                    if parts and getattr(parts[0], "text", ""):
                        return parts[0].text
                raise ValueError("응답에 text 없음")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                print(f"⚠️ Gemini 호출 실패 (시도 {attempt + 1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))


gem = GeminiClient()


# -----------------------
# 파일/계획 로딩
# -----------------------

def _latest_plan_path(mode: str) -> Optional[Path]:
    day_dir = Path(f"test_logs/{mode}/{_today()}")
    if day_dir.exists():
        # 가장 최신 *_plan.json
        cands = sorted(day_dir.glob("*_plan.json"))
        if cands:
            return cands[-1]
    root = Path(f"test_logs/{mode}")
    if not root.exists():
        return None
    all_plans = sorted(root.rglob("*_plan.json"))
    return all_plans[-1] if all_plans else None


def load_plan(plan: Optional[Dict[str, Any]] = None, plan_path: Optional[str | Path] = None, mode: str = DEF_MODE) -> Dict[str, Any]:
    if plan is not None:
        plan = dict(plan)
        plan.setdefault("meta", {}).setdefault("source_log", "(provided dict)")
        return plan
    if plan_path:
        p = Path(plan_path)
        if not p.exists():
            raise FileNotFoundError(f"plan 파일을 찾을 수 없습니다: {p}")
        return json.loads(p.read_text(encoding="utf-8"))
    p = _latest_plan_path(mode)
    if not p:
        raise FileNotFoundError("최신 *_plan.json을 찾을 수 없습니다. PlanAgent 실행을 먼저 진행하세요.")
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("meta", {}).setdefault("source_log", str(p))
    return data


# -----------------------
# JSON 강인 파싱
# -----------------------

def _parse_json(s: str) -> Optional[Dict[str, Any]]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    if start == -1:
        return None
    depth, end = 0, -1
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end != -1:
        try:
            return json.loads(s[start:end])
        except Exception:
            pass
    try:
        obj = ast.literal_eval(s[start:end] if end != -1 else s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


# -----------------------
# 텍스트 정리/검증
# -----------------------

FORBIDDEN_PATTERNS = [
    r"\b100%\b", r"무통증", r"완치", r"유일", r"최고", r"즉시\s*효과", r"파격", r"이벤트", r"특가",
    r"\d+\s*원", r"\d+\s*만원", r"가격", r"전화", r"\bTEL\b", r"http[s]?://", r"www\.",
]


def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    # 전각 괄호/이모지 등 과도한 특수기호 제거(관용적 최소)
    s = re.sub(r"[\u3000-\u303F\uFE10-\uFE1F\uFE30-\uFE4F]", "", s)
    return s


def _len_ok(title: str) -> bool:
    L = len(title)
    return 18 <= L <= 42  # 권장 22~38, 허용 폭 18~42


def _violates_forbidden(title: str) -> bool:
    t = title or ""
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def _contains_hospital(title: str, hospital_name: str) -> bool:
    if not hospital_name:
        return False
    # 병원명 그대로 혹은 공백 제거 매칭 방지
    name = re.escape(hospital_name)
    if re.search(name, title):
        return True
    compact = re.sub(r"\s+", "", hospital_name)
    return compact and compact in re.sub(r"\s+", "", title)


# # -----------------------
# # 프롬프트 로딩 (없으면 내장 텍스트 사용)
# # -----------------------

# EMBEDDED_GEN_PROMPT = """
# 당신은 치과 블로그 SEO 제목 전문가입니다. 단, 의료광고법을 위반하지 않도록 병원 홍보성 문구는 최소화합니다.
# 입력으로 주어지는 'plan' JSON(PlanAgent 결과)을 바탕으로,
# 병원명 비노출 원칙을 지키면서 지역 맥락(가능하면 자연스럽게 후반부),
# 카테고리 핵심, 페르소나 톤, 그리고 7개 섹션(서론/내원·방문/검사·진단/의료진 팁/치료 과정/체크포인트/마무리·결과)에 담긴 내용을 반영한
# 제목 후보 {N}개를 만들고 그중 최적 1개를 선택하세요.

# 제목 작성 규칙(의료광고 준수):
# 1) 한국어, 권장 22~38자. 과장/단정/치료효과 보장 표현 금지(예: 100%, 무통증, 완치, 유일/최고, 즉시 효과 등).
# 2) 카테고리 핵심 키워드 1개는 포함하되 나열 금지(예: “충치치료”, “신경치료”, “스케일링” 등에서 1개 선택).
# 3) 환자 관점에서 구체적·예상 이득이 드러나되 낚시 금지.
# 4) 병원명 직접 표기 금지. 지역 문구는 필요 시 문장 후반부에 자연스럽게(예: “— 서울 강남” 또는 “서울 강남에서”).
# 5) 전화/가격/이벤트/URL/내부링크 암시 금지. 불필요한 특수문자·이모지·전각괄호 지양.
# 6) 페르소나(tone_persona)가 있으면 각 후보의 어조/관점에 반영.

# 생성 지침:
# - 섹션 전개(서론→내원→검사→팁→치료→체크포인트→마무리) 가운데 핵심 포인트를 압축해 제목에서 기대 내용을 명확히 암시하세요.
# - [content_outline/summary]의 카테고리/증상/진료/치료 중 검색 의도에 가장 유효한 1개를 선택해 포함하세요.
# - 지역(city, district, region_phrase)이 있으면 맥락상 자연스러울 때만 제목 후반부에 덧붙이세요.
# - 후보마다 ‘angle’에 페르소나/섹션 반영 관점을 1문장으로 요약하세요(예: “관리형 페르소나: 사후관리/재내원 간격 강조”).

# 출력 JSON 스키마(정확히 이 구조 사용):
# {
#   "candidates": [
#     {"title":"...", "angle":"페르소나/섹션 반영 관점 요약"},
#     {"title":"...", "angle":"..."}
#   ],
#   "selected": {"title":"...", "why_best":"선정 이유(간단)"}
# }

# 입력 plan:
# """

# EMBEDDED_EVAL_PROMPT = """
# 다음은 치과 블로그 제목 후보들입니다. 규칙을 다시 확인하고 그중 최적 1개를 선택하세요.

# 선택 기준(요약):
# - 의료광고 준수(과장/단정/가격/전화/URL/병원명 금지)
# - 검색의도 적합성(카테고리/증상/진료/치료 중 1개 핵심키워드 포함, 나열 금지)
# - 명확성/구체성(환자 관점 기대이득 암시)
# - 지역 문구는 후반부 자연스러움(있을 때만)
# - 길이 적정(권장 22~38자 · 허용 18~42자)

# 출력 JSON 스키마:
# {
#   "selected": {"title":"...", "why_best":"선정 이유(간단)"}
# }

# 후보:
# """


def _load_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    # return fallback
    raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {path}")


# -----------------------
# 플랜→컨텍스트 추출
# -----------------------

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def extract_context(plan: Dict[str, Any]) -> Dict[str, Any]:
    ctx = {
        "hospital_name": _get(plan, "context_vars.hospital_name", ""),
        "category": _get(plan, "context_vars.category", ""),
        "city": _get(plan, "context_vars.city", ""),
        "district": _get(plan, "context_vars.district", ""),
        "region_phrase": _get(plan, "context_vars.region_phrase", ""),
        "must_include_one_of": _get(plan, "title_plan.must_include_one_of", []),
        "must_not_include": _get(plan, "title_plan.must_not_include", []),
        "guidance": _get(plan, "title_plan.guidance", ""),
        "tone": _get(plan, "title_plan.tone", ""),
        "representative_persona": plan.get("representative_persona", ""),
        "sections": _get(plan, "content_plan.sections", {}),
    }
    return ctx


def build_generation_prompt(plan: Dict[str, Any], N: int) -> str:
    # tpl = _load_text(GEN_PROMPT_PATH, EMBEDDED_GEN_PROMPT)
    tpl = _load_text(GEN_PROMPT_PATH)
    # plan JSON은 그대로 붙이되, 불필요 공백 축소
    plan_min = json.dumps(plan, ensure_ascii=False)
    return tpl.replace("{N}", str(N)).rstrip() + "\n" + plan_min


def build_evaluation_prompt(candidates_json: Dict[str, Any]) -> str:
    # tpl = _load_text(EVAL_PROMPT_PATH, EMBEDDED_EVAL_PROMPT)
    tpl = _load_text(EVAL_PROMPT_PATH)
    return tpl.rstrip() + "\n" + json.dumps(candidates_json, ensure_ascii=False)


# -----------------------
# 후보 생성 & 선택
# -----------------------

def generate_candidates(plan: Dict[str, Any], N: int = 5) -> Dict[str, Any]:
    prompt = build_generation_prompt(plan, N)
    sys_dir = (
        "You are an assistant that outputs ONLY valid JSON. No prose. No markdown fences.\n"
        "Return ONLY the JSON object per schema."
    )
    full_prompt = f"{sys_dir}\n\n{prompt}"
    raw = gem.generate(full_prompt)
    obj = _parse_json(raw) or {"candidates": [], "selected": {"title": "", "why_best": ""}}

    # 최소 스키마 보정
    if not isinstance(obj.get("candidates"), list):
        obj["candidates"] = []
    obj.setdefault("selected", {"title": "", "why_best": ""})

    # 정리/클린업
    hospital_name = _get(plan, "context_vars.hospital_name", "")
    cleaned: List[Dict[str, str]] = []
    for c in obj["candidates"]:
        title = _clean(str(c.get("title", "")))
        angle = _clean(str(c.get("angle", "")))
        if not title:
            continue
        if _contains_hospital(title, hospital_name):
            continue
        if _violates_forbidden(title):
            continue
        cleaned.append({"title": title, "angle": angle})

    obj["candidates"] = cleaned[:N]
    # selected는 평가 단계에서 확정
    obj["selected"] = {"title": "", "why_best": ""}
    return obj


def select_best(plan: Dict[str, Any], candidates_obj: Dict[str, Any]) -> Dict[str, Any]:
    # 길이/금지어 1차 필터링 + 너무 짧거나 긴 것은 제외
    hospital = _get(plan, "context_vars.hospital_name", "")
    filt = []
    for c in candidates_obj.get("candidates", []):
        t = c.get("title", "")
        if not _len_ok(t):
            continue
        if _contains_hospital(t, hospital):
            continue
        if _violates_forbidden(t):
            continue
        filt.append(c)

    cand_obj = {"candidates": filt or candidates_obj.get("candidates", [])}

    prompt = build_evaluation_prompt(cand_obj)
    sys_dir = (
        "You are an assistant that outputs ONLY valid JSON. No prose. No markdown fences.\n"
        "Return ONLY the JSON object per schema."
    )
    full = f"{sys_dir}\n\n{prompt}"
    raw = gem.generate(full)
    sel = _parse_json(raw) or {"selected": {"title": "", "why_best": ""}}

    # 보정: selected 누락 시 첫 후보 사용
    if not isinstance(sel.get("selected"), dict):
        sel["selected"] = {"title": "", "why_best": ""}
    if not sel["selected"].get("title") and cand_obj.get("candidates"):
        sel["selected"] = {"title": cand_obj["candidates"][0]["title"], "why_best": "최소 규칙 충족 및 명확성"}

    return sel


# -----------------------
# 저장
# -----------------------

def save_outputs(mode: str, result: Dict[str, Any], meta: Dict[str, Any]) -> Tuple[Path, Path]:
    out_dir = Path(f"test_logs/{mode}/{_today()}")
    _ensure_dir(out_dir)
    ts = _now()
    out_path = out_dir / f"{ts}_title.json"
    log_path = out_dir / f"{ts}_title_log.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, log_path


# -----------------------
# 메인 파이프라인
# -----------------------

def run(plan: Optional[Dict[str, Any]] = None, plan_path: Optional[str | Path] = None, mode: str = DEF_MODE, N: int = 5) -> Dict[str, Any]:
    plan_obj = load_plan(plan=plan, plan_path=plan_path, mode=mode)

    # 후보 생성
    cand_obj = generate_candidates(plan_obj, N=N)

    # 모델 선택
    sel_obj = select_best(plan_obj, cand_obj)

    # 최종 결과 스키마 조립
    final = {
        "candidates": cand_obj.get("candidates", []),
        "selected": sel_obj.get("selected", {"title": "", "why_best": ""}),
    }

    # 저장 메타
    meta = {
        "mode": mode,
        "timestamp": _now(),
        "plan_source": plan_obj.get("meta", {}).get("source_log", ""),
        "plan_snapshot": plan_obj,  # 추후 디버깅용
    }

    out, log = save_outputs(mode, final, meta)
    print(f"✅ Title 저장: {out}")
    print(f"🧾 로그 저장: {log}")
    return final

def run(plan: Optional[Dict[str, Any]] = None, plan_path: Optional[str | Path] = None, mode: str = DEF_MODE, N: int = 5) -> Dict[str, Any]:
    plan_obj = load_plan(plan=plan, plan_path=plan_path, mode=mode)

    # 후보 생성
    cand_obj = generate_candidates(plan_obj, N=N)

    # 모델 선택
    sel_obj = select_best(plan_obj, cand_obj)

    # 최종 결과 스키마 조립
    final = {
        "candidates": cand_obj.get("candidates", []),
        "selected": sel_obj.get("selected", {"title": "", "why_best": ""}),
    }

    # 사용 데이터 로그용 추출
    used_data = {
        "category": _get(plan_obj, "context_vars.category", ""),
        "city": _get(plan_obj, "context_vars.city", ""),
        "district": _get(plan_obj, "context_vars.district", ""),
        "region_phrase": _get(plan_obj, "context_vars.region_phrase", ""),
        "representative_persona": plan_obj.get("representative_persona", ""),
        "section_summaries": {k: v.get("summary", "") for k, v in _get(plan_obj, "content_plan.sections", {}).items()},
    }

    # 저장 메타
    meta = {
        "mode": mode,
        "timestamp": _now(),
        "plan_source": plan_obj.get("meta", {}).get("source_log", ""),
        "plan_snapshot": plan_obj,  # 추후 디버깅용
        "used_data": used_data,     # 제목 생성 시 참고한 주요 데이터
    }

    out, log = save_outputs(mode, final, meta)
    print(f"✅ Title 저장: {out}")
    print(f"🧾 로그 저장: {log}")
    print(f"📌 사용 데이터: {json.dumps(used_data, ensure_ascii=False, indent=2)}")
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TitleAgent — plan 기반 제목 후보 생성 및 최종 선택")
    parser.add_argument("--mode", default=DEF_MODE, choices=["test", "use"], help="로그 저장 모드")
    parser.add_argument("--plan", default="", help="plan JSON 경로(미지정 시 최신 파일 자동 탐색)")
    parser.add_argument("--num", type=int, default=5, help="제목 후보 개수")
    args = parser.parse_args()

    plan_path = args.plan if args.plan else None
    run(plan_path=plan_path, mode=args.mode, N=args.num)
