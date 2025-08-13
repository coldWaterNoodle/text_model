# agents/plan_agent.py
# -*- coding: utf-8 -*-
"""
PlanAgent (프롬프트 기반 · 7섹션 · 이미지 바인딩 · 스키마 리페어)
- 입력: input_agent 로그 최신 1건 또는 외부 dict
- 프롬프트: test_prompt/plan_generation_prompt.txt
- 모델: Gemini (GEMINI_API_KEY 필수 · 항상 호출) — JSON 파싱 실패 시 코드 기반 fallback
- 출력:
    - 본문: test_logs/{mode}/{YYYYMMDD}/{timestamp}_plan.json
    - 로그: test_logs/{mode}/{YYYYMMDD}/{timestamp}_plan_logs.json
- 호환:
    - Input 로그 파일명: *신규* {YYYYMMDD}_{HHMMSS}_input_logs.json (권장)
                         *구형* {YYYYMMDD}_input_log.json 도 자동 인식
"""

import os, json, re, ast, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# =======================
# 환경 & Gemini 클라이언트
# =======================
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")
genai.configure(api_key=API_KEY)


class GeminiClient:
    def __init__(self, model="models/gemini-1.5-flash", temperature=0.7, max_output_tokens=8192):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = 3
        self.retry_delay = 1.0

    def generate_text(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Gemini 텍스트 생성(재시도 포함)"""
        for attempt in range(self.max_retries):
            try:
                m = genai.GenerativeModel(self.model)
                resp = m.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=self.temperature if temperature is None else temperature,
                        max_output_tokens=self.max_output_tokens,
                        candidate_count=1,
                        top_p=0.95,
                        top_k=40,
                    )
                )
                if getattr(resp, "text", None):
                    return resp.text
                if getattr(resp, "candidates", None):
                    parts = getattr(resp.candidates[0].content, "parts", [])
                    if parts and getattr(parts[0], "text", ""):
                        return parts[0].text
                raise ValueError("응답에 text 없음")
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                print(f"⚠️ Gemini 호출 실패 (시도 {attempt+1}/{self.max_retries}): {e}")
                time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError("모든 재시도 실패")


gemini_client = GeminiClient()

# ===============
# 경로/시간 유틸
# ===============
PROMPT_PATH = Path("test_prompt/plan_generation_prompt.txt")

def _today() -> str:
    return datetime.now().strftime("%Y%m%d")

def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _safe_get(d: dict, path: str, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except Exception:
        return 0.0

# =======================================================
# 최신 input 로그 탐색 (신규/구형 파일명 모두 지원, 최신 1건 반환)
# =======================================================
def _latest_input_log(mode: str) -> Tuple[Optional[Path], Optional[dict]]:
    """
    우선순위:
      1) test_logs/{mode}/{YYYYMMDD}/*_input_logs.json (신규 규격)
      2) test_logs/{mode}/{YYYYMMDD}/*_input_log.json  (구형 규격)
      3) 상위 폴더 전체에서 위 두 패턴 중 최신 파일
    파일 내용이 배열이면 마지막 원소, dict면 그대로 반환
    """
    day_dir = Path(f"test_logs/{mode}/{_today()}")
    patterns_today = ["*_input_logs.json", "*_input_log.json"]
    for pat in patterns_today:
        hits = sorted(day_dir.glob(pat), key=_mtime, reverse=True)
        if hits:
            p = hits[0]
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return p, data[-1]
                if isinstance(data, dict):
                    return p, data
            except Exception:
                pass

    # 폴백: 모드 폴더 전체에서 최신 탐색
    root = Path(f"test_logs/{mode}")
    if not root.exists():
        return None, None
    hits = sorted(list(root.rglob("*_input_logs.json")) + list(root.rglob("*_input_log.json")),
                  key=_mtime, reverse=True)
    if not hits:
        return None, None
    p = hits[0]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return p, data[-1]
        if isinstance(data, dict):
            return p, data
    except Exception:
        return None, None
    return None, None

# ==================
# 텍스트 유틸
# ==================
def _compress(*texts: str, max_len: int = 280) -> str:
    joined = " ".join([t for t in texts if t]).strip()
    joined = re.sub(r"\s+", " ", joined)
    return joined[:max_len].rstrip()

# ==================
# 이미지 바인딩
# ==================
def _bind_images(section_key: str, row: dict) -> List[dict]:
    """
    신스키마 우선 + 구스키마 폴백
      - question3_visit_images | visit_images
      - question5_therapy_images | therapy_images
      - question7_result_images | result_images
    """
    mapping = {
        "2_visit":       [("question3_visit_images|visit_images", 2)],
        "3_inspection":  [("question3_visit_images|visit_images", 1)],
        "5_treatment":   [("question5_therapy_images|therapy_images", 3)],
        "6_check_point": [("question7_result_images|result_images", 1)],
        "7_conclusion":  [("question7_result_images|result_images", 1), ("hospital.business_card", 1)],
    }
    binds: List[dict] = []
    for src, limit in mapping.get(section_key, []):
        if src == "hospital.business_card":
            if _safe_get(row, "hospital.business_card"):
                binds.append({"from": "hospital.business_card", "position": "bottom"})
        else:
            keys = src.split("|")
            arr = []
            for k in keys:
                val = _safe_get(row, k, [])
                if isinstance(val, list) and val:
                    arr = val
                    break
            if arr:
                binds.append({"from": keys[0], "limit": min(limit, len(arr))})
    return binds

# ==========================
# 기본(Fallback) Plan 생성
# ==========================
def _fallback_plan(input_row: dict, mode: str) -> dict:
    city = _safe_get(input_row, "city", "")
    district = _safe_get(input_row, "district", "")
    region_phrase = _safe_get(input_row, "region_phrase", (city + " " + district).strip())
    hospital_name = _safe_get(input_row, "hospital.name", "")
    save_name = _safe_get(input_row, "hospital.save_name", "")
    category = _safe_get(input_row, "category", "")
    map_link = _safe_get(input_row, "hospital.map_link", "")

    q1 = _safe_get(input_row, "question1_concept", "")
    q2 = _safe_get(input_row, "question2_condition", "")
    q4 = _safe_get(input_row, "question4_treatment", "")
    q6 = _safe_get(input_row, "question6_result", "")
    q8 = _safe_get(input_row, "question8_extra", "")
    s  = _safe_get(input_row, "selected_symptom", "")
    p  = _safe_get(input_row, "selected_procedure", "")
    t  = _safe_get(input_row, "selected_treatment", "")
    teeth = ", ".join(_safe_get(input_row, "tooth_numbers", []) or [])

    title_plan = {
        "guidance": "지역+카테고리+핵심 키워드를 자연스럽게 포함(24~36자), 과장·단정·가격문구 금지",
        "must_include_one_of": ["{city}", "{district}", "{region_phrase}"],
        "must_include": [],
        "must_not_include": ["{hospital_name}", "가격", "이벤트", "전화번호"],
        "tone": "전문적·친절, 결과 단정 대신 개인차 암시",
        "hints": {"category": category, "region_examples": [city, district, region_phrase]}
    }

    sections_order = ["1_intro","2_visit","3_inspection","4_doctor_tip","5_treatment","6_check_point","7_conclusion"]
    sections = {
        "1_intro": {
            "subtitle": "왜 이 치료가 필요했나",
            "summary": _compress(f"[개념] {q1}", f"[증상] {q2}", f"[S/P/T] {s}/{p}/{t}"),
            "write_instruction": "content1_intro_prompt.txt",
            "must_include": ["{region_phrase}"],
            "may_include": ["{category}"],
            "must_not_include": [],
            "image_binding": _bind_images("1_intro", input_row)
        },
        "2_visit": {
            "subtitle": "내원 당시 상태와 상담 포인트",
            "summary": _compress(q2, "초진 상담·생활 불편·통증 유발 상황"),
            "write_instruction": "content2_visit_prompt.txt",
            "must_include": [],
            "may_include": ["{region_phrase}"],
            "must_not_include": [],
            "image_binding": _bind_images("2_visit", input_row)
        },
        "3_inspection": {
            "subtitle": "진단/검사 포인트",
            "summary": _compress(f"P(진료): {p}", f"치식: {teeth}"),
            "write_instruction": "content3_inspection_prompt.txt",
            "must_include": [],
            "may_include": ["{region_phrase}"],
            "must_not_include": [],
            "image_binding": _bind_images("3_inspection", input_row)
        },
        "4_doctor_tip": {
            "subtitle": "치과의사 한마디(선택/주의/대안)",
            "summary": _compress(q8, "대체옵션·주의사항·과장 금지"),
            "write_instruction": "content4_doctor_tip_prompt.txt",
            "must_include": [],
            "may_include": ["{hospital_name}"],
            "must_not_include": ["가격","이벤트"],
            "image_binding": _bind_images("4_doctor_tip", input_row)
        },
        "5_treatment": {
            "subtitle": "치료 과정과 재료 선택",
            "summary": _compress(q4, "재료·횟수·내원수·감염관리"),
            "write_instruction": "content5_treatment_prompt.txt",
            "must_include": [],
            "may_include": ["{category}"],
            "must_not_include": ["가격","무통증 단정"],
            "image_binding": _bind_images("5_treatment", input_row)
        },
        "6_check_point": {
            "subtitle": "체크포인트 & 관리법",
            "summary": _compress(q6, "재내원 기준·통증 변화 모니터링·가정관리"),
            "write_instruction": "content6_check_point_prompt.txt",
            "must_include": [],
            "may_include": ["{region_phrase}"],
            "must_not_include": ["과장표현"],
            "image_binding": _bind_images("6_check_point", input_row)
        },
        "7_conclusion": {
            "subtitle": "결론과 다음 단계",
            "summary": _compress("핵심 요점 회수", "정기검진/문의 안내(비상업적)", map_link),
            "write_instruction": "content7_conclusion_prompt.txt",
            "must_include": ["{hospital_name}"],
            "may_include": ["{region_phrase}"],
            "must_not_include": ["가격","이벤트","전화번호 직기재"],
            "image_binding": _bind_images("7_conclusion", input_row)
        }
    }

    return {
        "meta": {
            "mode": mode,
            "timestamp": _now(),
            "case_id": _safe_get(input_row, "case_id", ""),
            "source_log": input_row.get("source_log", "")
        },
        "context_vars": {
            "hospital_name": hospital_name,
            "save_name": save_name,
            "city": city,
            "district": district,
            "region_phrase": region_phrase,
            "category": category,
            "map_link": map_link
        },
        "title_plan": title_plan,
        "content_plan": {
            "sections_order": sections_order,
            "sections": sections
        }
    }

# ===========================
# 프롬프트 로딩 & 안전 치환
# ===========================
def _load_prompt() -> str:
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {PROMPT_PATH}")
    return PROMPT_PATH.read_text(encoding="utf-8")

def _render_prompt(tpl: str, row: dict) -> str:
    """명시 변수만 {var}→값 치환. 다른 {중괄호}는 보존."""
    vars_map = {
        "hospital_name": _safe_get(row, "hospital.name", ""),
        "save_name": _safe_get(row, "hospital.save_name", ""),
        "city": _safe_get(row, "city", ""),
        "district": _safe_get(row, "district", ""),
        "region_phrase": (_safe_get(row, "region_phrase", "") or f"{_safe_get(row,'city','')} {_safe_get(row,'district','')}".strip()).strip(),
        "category": _safe_get(row, "category", ""),
        "selected_symptom": _safe_get(row, "selected_symptom", ""),
        "selected_procedure": _safe_get(row, "selected_procedure", ""),
        "selected_treatment": _safe_get(row, "selected_treatment", ""),
        "tooth_numbers": ", ".join(_safe_get(row, "tooth_numbers", []) or []),
        "question1_concept": _safe_get(row, "question1_concept", ""),
        "question2_condition": _safe_get(row, "question2_condition", ""),
        "question4_treatment": _safe_get(row, "question4_treatment", ""),
        "question6_result": _safe_get(row, "question6_result", ""),
        "question8_extra": _safe_get(row, "question8_extra", ""),
        "representative_persona": _safe_get(row, "representative_persona", ""),
        "clinical_context": json.dumps(_safe_get(row, "clinical_context", {}), ensure_ascii=False),
    }
    # 이중 중괄호 보호
    L, R = "§§__L__§§", "§§__R__§§"
    work = tpl.replace("{{", L).replace("}}", R)

    # {키}만 치환
    pattern = re.compile(r"\{(" + "|".join(map(re.escape, vars_map.keys())) + r")\}")
    work = pattern.sub(lambda m: str(vars_map.get(m.group(1), "")), work)

    return work.replace(L, "{{").replace(R, "}}")

# ===================
# LLM & JSON 파싱
# ===================
def _call_llm(prompt: str) -> str:
    sys_dir = (
        "You are a planning assistant. "
        "Output ONLY valid JSON. No prose. No markdown fences."
    )
    full_prompt = f"{sys_dir}\n\n{prompt}\n\nReturn only JSON."
    return gemini_client.generate_text(full_prompt)

def _try_json_load(s: str) -> Optional[dict]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    # 코드블록 제거
    s = re.sub(r"^```(json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()

    # 1차: 그대로
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2차: 첫 '{' ~ 균형 '}' 추출 후 시도
    start = s.find("{")
    if start != -1:
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

    # 3차: 파이썬 dict 스타일
    try:
        obj = ast.literal_eval(s[start:end] if start != -1 and end != -1 else s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

# =================
# 스키마 리페어
# =================
REQUIRED_SECTIONS = ["1_intro","2_visit","3_inspection","4_doctor_tip","5_treatment","6_check_point","7_conclusion"]

def _repair_plan(obj: dict, input_row: dict, mode: str) -> dict:
    """필수 키, 섹션 7개 보장 + 이미지 바인딩 보강"""
    if not isinstance(obj, dict):
        return _fallback_plan(input_row, mode)

    # title_plan
    tp = obj.get("title_plan") or {}
    tp.setdefault("guidance", "지역+카테고리+핵심 키워드/24~36자/과장금지")
    tp.setdefault("must_include_one_of", ["{city}", "{district}", "{region_phrase}"])
    tp.setdefault("must_include", [])
    tp.setdefault("must_not_include", ["{hospital_name}", "가격", "이벤트", "전화번호"])
    tp.setdefault("tone", "전문적·친절")
    obj["title_plan"] = tp

    # content_plan
    cp = obj.get("content_plan") or {}
    cp["sections_order"] = REQUIRED_SECTIONS
    sections = cp.get("sections") or {}

    def ensure_section(k: str, default_sub: str, default_sum: str, prompt_file: str,
                       must=None, may=None, must_not=None) -> dict:
        s = sections.get(k) or {}
        s.setdefault("subtitle", default_sub)
        s.setdefault("summary", default_sum)
        s.setdefault("write_instruction", prompt_file)
        s.setdefault("must_include", must or [])
        s.setdefault("may_include", may or [])
        s.setdefault("must_not_include", must_not or [])
        s["image_binding"] = _bind_images(k, input_row)
        return s

    city = _safe_get(input_row, "city", "")
    district = _safe_get(input_row, "district", "")
    region_phrase = _safe_get(input_row, "region_phrase", (city + " " + district).strip())
    hospital_name = _safe_get(input_row, "hospital.name", "")
    category = _safe_get(input_row, "category", "")

    q1 = _safe_get(input_row, "question1_concept", "")
    q2 = _safe_get(input_row, "question2_condition", "")
    q4 = _safe_get(input_row, "question4_treatment", "")
    q6 = _safe_get(input_row, "question6_result", "")
    q8 = _safe_get(input_row, "question8_extra", "")
    spt = f"{_safe_get(input_row,'selected_symptom','')}/{_safe_get(input_row,'selected_procedure','')}/{_safe_get(input_row,'selected_treatment','')}"
    teeth = ", ".join(_safe_get(input_row, "tooth_numbers", []) or [])

    sections["1_intro"] = ensure_section(
        "1_intro","왜 이 치료가 필요했나",
        _compress(f"[개념] {q1}", f"[증상] {q2}", f"[S/P/T] {spt}"),
        "content1_intro_prompt.txt",
        must=["{region_phrase}"], may=["{category}"], must_not=[]
    )
    sections["2_visit"] = ensure_section(
        "2_visit","내원 당시 상태와 상담 포인트",
        _compress(q2, "초진 상담·생활 불편·통증 유발 상황"),
        "content2_visit_prompt.txt",
        must=[], may=["{region_phrase}"], must_not=[]
    )
    sections["3_inspection"] = ensure_section(
        "3_inspection","진단/검사 포인트",
        _compress(f"P(진료): {_safe_get(input_row,'selected_procedure','')}", f"치식: {teeth}"),
        "content3_inspection_prompt.txt",
        must=[], may=["{region_phrase}"], must_not=[]
    )
    sections["4_doctor_tip"] = ensure_section(
        "4_doctor_tip","치과의사 한마디(선택/주의/대안)",
        _compress(q8, "대체옵션·주의사항·과장 금지"),
        "content4_doctor_tip_prompt.txt",
        must=[], may=["{hospital_name}"], must_not=["가격","이벤트"]
    )
    sections["5_treatment"] = ensure_section(
        "5_treatment","치료 과정과 재료 선택",
        _compress(q4, "재료·횟수·내원수·감염관리"),
        "content5_treatment_prompt.txt",
        must=[], may=["{category}"], must_not=["가격","무통증 단정"]
    )
    sections["6_check_point"] = ensure_section(
        "6_check_point","체크포인트 & 관리법",
        _compress(q6, "재내원 기준·통증 변화 모니터링·가정관리"),
        "content6_check_point_prompt.txt",
        must=[], may=["{region_phrase}"], must_not=["과장표현"]
    )
    map_link = _safe_get(input_row, "hospital.map_link", "")
    sections["7_conclusion"] = ensure_section(
        "7_conclusion","결론과 다음 단계",
        _compress("핵심 요점 회수", "정기검진/문의 안내(비상업적)", map_link),
        "content7_conclusion_prompt.txt",
        must=["{hospital_name}"], may=["{region_phrase}"], must_not=["가격","이벤트","전화번호 직기재"]
    )

    cp["sections"] = sections
    obj["content_plan"] = cp

    # context_vars
    obj.setdefault("context_vars", {
        "hospital_name": hospital_name,
        "save_name": _safe_get(input_row, "hospital.save_name", ""),
        "city": city,
        "district": district,
        "region_phrase": region_phrase,
        "category": category,
        "map_link": map_link
    })

    # meta
    obj.setdefault("meta", {})
    obj["meta"].setdefault("mode", mode)
    obj["meta"].setdefault("timestamp", _now())
    obj["meta"].setdefault("case_id", _safe_get(input_row, "case_id", ""))
    obj["meta"].setdefault("source_log", input_row.get("source_log", ""))

    return obj

# ================
# 저장 & 실행
# ================
def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def save_plan(plan: dict, mode: str) -> Path:
    out_dir = Path(f"test_logs/{mode}/{_today()}")
    _ensure_dir(out_dir)
    path = out_dir / f"{_now()}_plan.json"
    _save_json(path, plan)
    return path

def save_plan_log(log_payload: dict, mode: str) -> Path:
    out_dir = Path(f"test_logs/{mode}/{_today()}")
    _ensure_dir(out_dir)
    path = out_dir / f"{_now()}_plan_logs.json"
    _save_json(path, log_payload)
    return path

def main(mode: str = "use", input_data: Optional[dict] = None):
    # 입력 확보
    if input_data is None:
        src_path, row = _latest_input_log(mode)
        if row is None:
            print("⚠️ 최신 input_log를 찾지 못했습니다. 먼저 input_agent를 실행하세요.")
            return None
        row["source_log"] = str(src_path)
    else:
        row = dict(input_data)
        row["source_log"] = "(provided dict)"

    # 프롬프트 호출 → JSON 파싱 → 리페어
    plan_obj: dict
    llm_text: str = ""
    prompt_rendered: str = ""
    success = True
    error_msg = ""

    try:
        print("🔄 Gemini: plan 생성 중 ...")
        tpl = _load_prompt()
        prompt_rendered = _render_prompt(tpl, row)
        llm_text = _call_llm(prompt_rendered)
        parsed = _try_json_load(llm_text)
        if parsed is None:
            raise ValueError("LLM JSON 파싱 실패")
        plan_obj = _repair_plan(parsed, row, mode)
        print("✅ Gemini 계획 생성 성공")
    except Exception as e:
        success = False
        error_msg = str(e)
        print(f"⚠️ LLM 생성 또는 파싱 실패, fallback 사용: {e}")
        plan_obj = _fallback_plan(row, mode)

    # 저장
    plan_path = save_plan(plan_obj, mode)

    # 로그 저장: test_logs/{mode}/{YYYYMMDD}/{timestamp}_plan_logs.json
    log_payload = {
        "meta": {
            "mode": mode,
            "timestamp": _now(),
            "model": gemini_client.model,
            "temperature": gemini_client.temperature,
            "max_output_tokens": gemini_client.max_output_tokens,
            "success": success,
            "error": error_msg,
        },
        "input": {
            "source_log_path": row.get("source_log", ""),
            "case_id": _safe_get(row, "case_id", ""),
        },
        "prompt": prompt_rendered,
        "llm_raw": llm_text,
        "output_paths": {
            "plan_path": str(plan_path),
        },
        "plan_preview": {
            "title_plan": plan_obj.get("title_plan", {}),
            "sections_order": plan_obj.get("content_plan", {}).get("sections_order", []),
        }
    }
    log_path = save_plan_log(log_payload, mode)

    print(f"✅ plan 저장: {plan_path}")
    print(f"📝 로그 저장: {log_path}")
    return plan_obj

if __name__ == "__main__":
    mode = input("모드 선택 (기본 use, test / use) : ").strip().lower() or "use"
    main(mode=mode)
