# agents/plan_agent.py
# -*- coding: utf-8 -*-

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json, re
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

# =========================
# 환경 & 모델
# =========================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel(model_name="models/gemini-1.5-flash")

# 외부 의존 (InputAgent)
try:
    from agents.input_agent_old2 import InputAgent  # noqa
except Exception:
    InputAgent = None

# =========================
# 상수/경로
# =========================
GEN_PROMPT_PATH = Path("test_prompt/plan_generation_prompt.txt")
EVAL2_PROMPT_PATH = Path("test_prompt/plan_evaluation_prompt.txt")
NWAY_PROMPT_PATH = Path("test_prompt/plan_nway_evaluation_prompt.txt")
TEST_DIR = Path("test_logs/test")
USE_DIR = Path("test_logs/use")
CLI_DIR = Path("test_logs/cli")
TEST_IMAGE_DIR = Path("test_data/test_image")
HOSPITAL_IMAGE_DIR = Path("test_data/hospital_image")

# 섹션 ID/타이틀(고정 구분자) + 표시 순서 (계획용 7섹션)
SECTION_ORDER = [
    (1, "1_intro",        "서론"),
    (2, "2_visit",        "내원/방문"),
    (3, "3_inspection",   "검사·진단"),
    (4, "4_doctor_tip",   "의료진 팁"),
    (5, "5_treatment",    "치료 과정"),
    (6, "6_check_point",  "체크포인트"),
    (7, "7_conclusion",   "마무리/결과"),
]
SECTION_ID_BY_NO = {no: sid for no, sid, _ in SECTION_ORDER}
SECTION_TITLES = {sid: title for _, sid, title in SECTION_ORDER}

# 이미지 설명 → 섹션 배치 키워드 (계획용)
KEYWORDS_TO_SECTION = [
    (2, ["내원", "초진", "방문", "접수", "상담", "첫 방문", "문진", "대기"]),
    (3, ["검사", "진단", "ct", "엑스레이", "x-ray", "파노라마", "3d", "스캔", "측정", "영상"]),
    (4, ["팁", "주의", "생활", "관리법", "의사", "설명", "조언", "가이드", "faq"]),
    (5, ["치료", "시술", "과정", "수복", "발치", "스케일링", "레진", "임플란트", "크라운", "근관", "세척"]),
    (6, ["체크", "유지", "관리", "리콜", "주의사항", "내원 간격"]),
    (7, ["결과", "후기", "전후", "예후", "변화", "완료", "재내원", "만족"]),
]

# =========================
# 유틸
# =========================
class SafeDict(dict):
    """템플릿 포맷에서 키가 없어도 빈 문자열로 안전 치환"""
    def __missing__(self, key):  # noqa
        return ""

def _now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _latest_input(dir_path: Path) -> Optional[Path]:
    files = list(dir_path.glob("*_input_log.json"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def _read_text(path: Path, default: str = "") -> str:
    return path.read_text(encoding="utf-8") if path.exists() else default

def _clean_fenced(text: str) -> str:
    lines = (text or "").splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _gen(prompt: str) -> str:
    raw = model.generate_content(prompt).text or ""
    return _clean_fenced(raw)

def _extract_json(output: str) -> dict:
    txt = _clean_fenced(output)
    try:
        return json.loads(txt)
    except Exception:
        start = txt.find("{")
        if start == -1:
            return {}
        stack = 0; end = -1
        for i, ch in enumerate(txt[start:], start):
            if ch == "{": stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    end = i + 1; break
        if end == -1:
            return {}
        try:
            return json.loads(txt[start:end])
        except Exception:
            return {}

def _extract_region(h: Dict[str, Any]) -> Tuple[str, str]:
    city = (h or {}).get("city", "") or ""
    district = (h or {}).get("district", "") or ""
    if city or district:
        return city, district
    addr = (h or {}).get("address", "") or ""
    toks = addr.split()
    c, d = "", ""
    if toks:
        c = toks[0].replace("특별시","").replace("광역시","").replace("도","")
        for tk in toks[1:4]:
            if re.search(r"(구|시|군)$", tk):
                d = tk; break
    return c, d

def _safe(obj, typ, default):
    return obj if isinstance(obj, typ) else default

def _safe_list(x):
    return x if isinstance(x, list) else []

def _mk_image_ref_list(input_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """images_index에 있는 ref 포맷을 그대로 계획에 사용"""
    idx = _safe(input_data.get("images_index"), dict, {})
    return {
        "visit_refs": _safe_list(idx.get("visit_refs")),
        "therapy_refs": _safe_list(idx.get("therapy_refs")),
        "result_refs": _safe_list(idx.get("result_refs")),
    }

def _pick_section_by_desc(desc: str, default_sec: int) -> int:
    d = (desc or "").lower()
    for sec, kws in KEYWORDS_TO_SECTION:
        for kw in kws:
            if kw in d:
                return sec
    return default_sec

def _place_images_plan(input_data: Dict[str, Any]) -> Dict[int, List[str]]:
    """
    계획용 배치: ref("visit_images:0" 등) 기준으로 섹션 번호에 매핑
    - visit → 기본 2
    - therapy → 기본 5
    - result → 기본 7
    설명 키워드가 있으면 재배치 (2~7)
    """
    md: Dict[int, List[str]] = {i: [] for i in range(1, 8)}
    refs = _mk_image_ref_list(input_data)

    # visit_images
    for i, r in enumerate(refs["visit_refs"]):
        desc = ""
        if i < len(_safe_list(input_data.get("visit_images"))):
            desc = input_data["visit_images"][i].get("description","")
        sec = _pick_section_by_desc(desc, 2)
        md[sec].append(r)

    # therapy_images
    for i, r in enumerate(refs["therapy_refs"]):
        desc = ""
        if i < len(_safe_list(input_data.get("therapy_images"))):
            desc = input_data["therapy_images"][i].get("description","")
        sec = _pick_section_by_desc(desc, 5)
        md[sec].append(r)

    # result_images
    for i, r in enumerate(refs["result_refs"]):
        desc = ""
        if i < len(_safe_list(input_data.get("result_images"))):
            desc = input_data["result_images"][i].get("description","")
        sec = _pick_section_by_desc(desc, 7)
        md[sec].append(r)

    return md

# =========================
# 키워드 유틸
# =========================
TOKEN_RE = re.compile(r"[^가-힣A-Za-z0-9]+")

def _keywords_from_text(s: str, max_n: int = 12) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    toks = [w for w in TOKEN_RE.sub(" ", s).split() if len(w) >= 2]
    out, seen = [], set()
    for w in toks:
        lw = w.lower()
        if lw not in seen:
            seen.add(lw)
            out.append(w)
        if len(out) >= max_n:
            break
    return out

# =========================
# 프롬프트 컨텍스트 (모든 키 사전 준비)
# =========================
def _build_prompt_context(input_data: dict) -> dict:
    d = input_data or {}
    h = _safe(d.get("hospital"), dict, {})
    city, district = _extract_region(h)

    geo = _safe(d.get("geo_branding"), dict, {})
    region_phrase = (h.get("region_phrase") or "").strip() or (geo.get("region_line") or "").strip()
    if not region_phrase:
        region_phrase = f"{city} {district}".strip()
    clinic_alias = (geo.get("clinic_alias") or h.get("name") or "").strip()
    region_line = (geo.get("region_line") or region_phrase).strip()

    sym = d.get("selected_symptom","")
    proc = d.get("selected_procedure","")
    tx = d.get("selected_treatment","")
    sym_kw = _keywords_from_text(sym); proc_kw = _keywords_from_text(proc); tx_kw = _keywords_from_text(tx)

    must = _safe(d.get("must_include_facts"), dict, {})
    equip_kw = _safe_list(must.get("equip"))
    tooth_fdi = _safe_list(must.get("tooth_fdi"))

    ctx = {
        # 병원(표시 제한 필드만 계획에 사용)
        "hospital_name": h.get("name",""),
        "hospital_save_name": h.get("save_name",""),
        "hospital_address": h.get("address",""),
        "hospital_city": city,
        "hospital_district": district,
        "hospital_region_phrase": region_phrase,
        "hospital_homepage": h.get("homepage",""),
        "hospital_map_link": h.get("map_link",""),
        "hospital_logo": h.get("logo",""),
        "hospital_business_card": h.get("business_card",""),
        "clinic_alias": clinic_alias,
        "region_line": region_line,
        "region_phrase": region_phrase,
        "hospital_city_district": f"{city} {district}".strip(),
        "homepage_url": h.get("homepage",""),
        "map_url": h.get("map_link",""),
        "hospital_logo_path": (HOSPITAL_IMAGE_DIR / h.get("logo","")).as_posix() if h.get("logo") else "",
        "hospital_business_card_path": (HOSPITAL_IMAGE_DIR / h.get("business_card","")).as_posix() if h.get("business_card") else "",

        # 분류/선택
        "category": d.get("category",""),
        "selected_symptom": sym,
        "selected_procedure": proc,
        "selected_treatment": tx,
        "symptom": sym, "procedure": proc, "treatment": tx,

        # 키워드
        "symptoms_keywords": ", ".join(sym_kw),
        "procedures_keywords": ", ".join(proc_kw),
        "treatments_keywords": ", ".join(tx_kw),
        "symptoms_keywords_list_json": json.dumps(sym_kw, ensure_ascii=False),
        "procedures_keywords_list_json": json.dumps(proc_kw, ensure_ascii=False),
        "treatments_keywords_list_json": json.dumps(tx_kw, ensure_ascii=False),

        # Q1~Q8
        "question1_concept": d.get("question1_concept",""),
        "question2_condition": d.get("question2_condition",""),
        "question4_treatment": d.get("question4_treatment",""),
        "question6_result": d.get("question6_result",""),
        "question8_extra": d.get("question8_extra",""),
        "q1": d.get("question1_concept",""),
        "q2": d.get("question2_condition",""),
        "q4": d.get("question4_treatment",""),
        "q6": d.get("question6_result",""),
        "q8": d.get("question8_extra",""),

        # 페르소나
        "persona_candidates": d.get("persona_candidates", []),
        "selected_personas": d.get("selected_personas", []),
        "representative_persona": d.get("representative_persona",""),
        "persona_structure_guide": d.get("persona_structure_guide",""),

        # 기타
        "content_flow_hint": d.get("content_flow_hint",""),
        "link_policy": d.get("link_policy", {"homepage_in_body_once": True, "map_in_footer_only": True}),
        "link_policy_homepage_in_body_once": bool(_safe(d.get("link_policy"), dict, {}).get("homepage_in_body_once", True)),
        "link_policy_map_in_footer_only": bool(_safe(d.get("link_policy"), dict, {}).get("map_in_footer_only", True)),
        "geo_branding": geo,
        "meta_panel": d.get("meta_panel", {}),
        "treatment_period": _safe(d.get("meta_panel"), dict, {}).get("treatment_period",""),
        "visit_images": d.get("visit_images", []),
        "therapy_images": d.get("therapy_images", []),
        "result_images": d.get("result_images", []),
        "images_index": d.get("images_index", {}),
        "visit_images_count": len(_safe_list(d.get("visit_images"))),
        "therapy_images_count": len(_safe_list(d.get("therapy_images"))),
        "result_images_count": len(_safe_list(d.get("result_images"))),

        "must_include_facts": d.get("must_include_facts", {}),
        "equip_keywords": ", ".join(equip_kw),
        "equip_keywords_list_json": json.dumps(equip_kw, ensure_ascii=False),
        "tooth_fdi_join": ", ".join(tooth_fdi),

        "mode": d.get("mode",""),
        "schema_version": d.get("schema_version",""),
    }

    # 별칭 키
    ctx["city"] = ctx["hospital_city"]; ctx["district"] = ctx["hospital_district"]
    ctx["address"] = ctx["hospital_address"]; ctx["homepage"] = ctx["hospital_homepage"]; ctx["map_link"] = ctx["hospital_map_link"]
    ctx["hospital"] = h

    # JSON 문자열 버전 (프롬프트에서 그대로 쓰고 싶을 때)
    ctx["visit_images_json"] = json.dumps(ctx["visit_images"], ensure_ascii=False)
    ctx["therapy_images_json"] = json.dumps(ctx["therapy_images"], ensure_ascii=False)
    ctx["result_images_json"] = json.dumps(ctx["result_images"], ensure_ascii=False)
    ctx["images_index_json"] = json.dumps(ctx["images_index"], ensure_ascii=False)
    ctx["persona_candidates_json"] = json.dumps(ctx["persona_candidates"], ensure_ascii=False)
    ctx["selected_personas_json"] = json.dumps(ctx["selected_personas"], ensure_ascii=False)
    ctx["geo_branding_json"] = json.dumps(ctx["geo_branding"], ensure_ascii=False)
    ctx["meta_panel_json"] = json.dumps(ctx["meta_panel"], ensure_ascii=False)
    ctx["must_include_facts_json"] = json.dumps(ctx["must_include_facts"], ensure_ascii=False)
    ctx["link_policy_json"] = json.dumps(ctx["link_policy"], ensure_ascii=False)

    # None 방어
    for k in list(ctx.keys()):
        if ctx[k] is None:
            ctx[k] = "" if not isinstance(ctx[k], (dict, list)) else ({} if isinstance(ctx[k], dict) else [])

    return ctx

# =========================
# 컴플라이언스 린터 (기본 OFF) — 계획 텍스트(부제/아웃라인)만 검사
# =========================
BANNED_PATTERNS = [
    r"100 ?%", r"무통증", r"완치", r"영구적", r"평생보장", r"유일(무|한)?", r"최고(의)?",
    r"즉시 ?효과", r"보장(됩니다|해드립니다)?", r"전혀 ?아프지", r"확실(히|한) ?치료"
]
PHONE_RE = re.compile(r"\b0\d{1,2}-\d{3,4}-\d{4}\b")
URL_RE = re.compile(r"https?://\S+")
DETAILED_ADDR_RE = re.compile(r"\b(\d{1,3}-\d{1,3}|\d{1,3}동|\d{1,3}로|\d{1,3}길|\d{1,4}번지)\b")

def _lint_plan(plan: Dict[str, Any], enabled: bool = False) -> Dict[str, Any]:
    report = {"enabled": enabled, "sections": {}}
    if not enabled:
        return report
    def scan(text: str) -> Dict[str, Any]:
        flags = []
        for pat in BANNED_PATTERNS:
            if re.search(pat, text or ""):
                flags.append(pat)
        phones = PHONE_RE.findall(text or "")
        urls = URL_RE.findall(text or "")
        addrs = DETAILED_ADDR_RE.findall(text or "")
        return {
            "banned_patterns": flags,
            "phones": phones,
            "urls": urls,
            "detailed_addr_hits": len(addrs),
        }
    for sid, sec in plan.get("content_plan", {}).get("sections", {}).items():
        text = " ".join([
            " ".join(_safe_list(sec.get("subtitle_suggestions"))),
            " ".join(_safe_list(sec.get("content_outline"))),
            " ".join(_safe_list(sec.get("must_include")))
        ])
        report["sections"][sid] = scan(text)
    return report

# =========================
# 계획 폴백: 모델 JSON이 비거나 7섹션 미만일 때 안전 생성
# =========================
def _fallback_plan(ctx: dict, image_placements: Dict[int, List[str]]) -> Dict[str, Any]:
    # 섹션별 목적/핵심 포함 포인트를 얇게 제안
    def points(*xs): return [x for x in xs if x]
    def img_refs(no: int) -> List[str]: return image_placements.get(no, [])

    content_sections = {}
    for no, sid, ko in SECTION_ORDER:
        content_sections[sid] = {
            "title": ko,
            "purpose": {
                "1_intro": "핵심 메시지와 글의 흐름 안내",
                "2_visit": "내원 배경과 초기 증상/문진",
                "3_inspection": "검사 방법과 진단 근거",
                "4_doctor_tip": "생활관리/주의/FAQ형 팁",
                "5_treatment": "치료 선택 근거와 절차·재료",
                "6_check_point": "주의점·유지관리·재내원 간격",
                "7_conclusion": "요약/개인차/다음 단계 안내"
            }.get(sid, ""),
            "subtitle_suggestions": [
                f"{ko} | {ctx.get('category','')}".strip(" |")
            ],
            "content_outline": points(
                f"[카테고리] {ctx.get('category','')}",
                f"[증상] {ctx.get('selected_symptom','')}",
                f"[진료] {ctx.get('selected_procedure','')}",
                f"[치료] {ctx.get('selected_treatment','')}",
                f"[Q1] {ctx.get('q1','')}",
                f"[Q2] {ctx.get('q2','')}" if no <= 3 else "",
                f"[Q4] {ctx.get('q4','')}" if no in (4,5) else "",
                f"[Q6] {ctx.get('q6','')}" if no in (6,7) else "",
                f"[Q8] {ctx.get('q8','')}"
            ),
            "must_include": _safe_list(ctx.get("must_include_facts", {}).get("tooth_fdi")) + _safe_list(ctx.get("must_include_facts", {}).get("equip")),
            "image_refs": img_refs(no),
            "tone_persona": ctx.get("representative_persona",""),
            "compliance_notes": [
                "과장/단정 금지(100%, 무통증 등)",
                "개인차 고지",
                "전화/가격/이벤트/내부 링크 금지",
                "병원 정보는 푸터에서만(이름/지역/홈페이지/지도)"
            ]
        }

    plan = {
        "title_plan": {
            "rules": [
                "검색 의도 부합, 24~36자 내외",
                "병원명 미포함, 지역 키워드 자연스러운 포함",
                "과장/단정 표현 금지, 개인차 암시 가능",
            ],
            "must_include_hints": [
                f"지역: {ctx.get('hospital_city','')} {ctx.get('hospital_district','')}".strip(),
                f"카테고리: {ctx.get('category','')}",
            ],
            "avoid": ["전화번호/이벤트/가격", "유일·최고·무통증 등 표현"]
        },
        "content_plan": {
            "sections_order": [sid for _, sid, _ in SECTION_ORDER],
            "sections": content_sections
        },
        "assets_plan": {
            "use_logo": False,
            "business_card_position": "bottom",
            "business_card_file": (Path(ctx.get("hospital_business_card_path")).name if ctx.get("hospital_business_card_path") else "")
        },
        "meta_footer_plan": {
            "expose_fields": ["name","region(city,district)","homepage","map_link"]
        },
        "image_placements": image_placements
    }
    return plan

# =========================
# 본체 클래스
# =========================
class PlanAgent:
    """
    PlanAgent generates blog plans using templates and selects the best candidate.
    - Test mode: 사용자 입력 X (input data : 최근 CLI log 읽어와 사용, test_log/cli) -> 2-way 평가(evaluation) 진행
    - Use mode: 사용자 입력 O (input_data : 직접 CLI 입력) → N-way 평가(evaluation) 진행
    - 첫 번째 코드의 완전한 데이터 처리 + 두 번째 코드의 깔끔한 구조

    Returns:
    - result_dict: Parsed JSON from best_output 
    - candidates: List of all candidate JSON strings
    - evaluation: Dict containing 'selected' index and 'reason' details
    - input_data: The dict used for generation
    """
    def __init__(
        self,
        gen_template_path: str = "test_prompt/plan_generation_prompt.txt",
        eval2_template_path: str = "test_prompt/plan_evaluation_prompt.txt", 
        nway_template_path: str = "test_prompt/plan_nway_evaluation_prompt.txt",
        default_nway_rounds: int = 3,
        lint_mode: bool = False
    ):
        self.gen_template = self._load_template(gen_template_path, "생성")
        self.eval2_template = self._load_template(eval2_template_path, "2-way 평가")
        self.nway_template = self._load_template(nway_template_path, "N-way 평가")
        self.default_nway_rounds = default_nway_rounds
        self.lint_mode = bool(lint_mode)

    def _load_template(self, path: str, name: str) -> str:
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"{name} 프롬프트 파일을 찾을 수 없습니다: {file}")
        return file.read_text(encoding="utf-8")

    def _prepare_input(self, input_data: Optional[dict], mode: str, source: str) -> Tuple[dict, str]:
        """입력 데이터 준비 - 첫 번째 코드 로직 적용"""
        if input_data is not None:
            return input_data, "(직접 제공)"
            
        if mode not in ("test", "use", "cli"):
            raise ValueError("mode는 'test', 'use', 또는 'cli' 여야 합니다.")
            
        # CLI 호환성을 위한 처리
        if mode == "cli":
            base = CLI_DIR
        else:
            base = TEST_DIR if mode == "test" else USE_DIR
            
        base.mkdir(parents=True, exist_ok=True)
        
        if source == "latest":
            ipath = _latest_input(base)
            if ipath is None:
                raise FileNotFoundError(f"{base} 에 최신 *_input_log.json 이 없습니다.")
            with open(ipath, "r", encoding="utf-8") as f:
                return json.load(f), str(ipath)
        elif source == "collect":
            if InputAgent is None:
                raise RuntimeError("InputAgent 를 불러올 수 없습니다.")
            ia = InputAgent()
            data = ia.run_test() if mode == "test" else ia.run_use()
            return data, "(via InputAgent)"
        else:
            raise ValueError("source는 'latest' 또는 'collect' 여야 합니다.")

    def _generate_candidates(self, prompt: str, rounds: int) -> List[str]:
        """후보 생성"""
        candidates: List[str] = []
        for i in range(rounds):
            raw = model.generate_content(prompt).text.strip()
            output = _clean_fenced(raw)
            if output:
                candidates.append(output)
            else:
                print(f"⚠️ 후보 {i+1} 빈 응답")
        if len(candidates) < 2:
            raise ValueError(f"후보가 부족합니다: {len(candidates)}")
        return candidates

    def evaluate_candidates(self, candidates: List[str]) -> Tuple[str, str, dict]:
        """2-way 평가"""
        prompt = self.eval2_template.format(
            candidate_1=candidates[0],
            candidate_2=candidates[1]
        )
        raw = model.generate_content(prompt).text.strip()
        eval_text = _clean_fenced(raw)
        eval_result = json.loads(eval_text)
        sel = eval_result.get("selected", "").strip()
        reason = eval_result.get("reason", {})
        return candidates[int(sel[-1]) - 1], sel, reason

    def evaluate_candidates_nway(self, candidates: List[str]) -> Tuple[str, str, dict]:
        """N-way 평가"""
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
        eval_text = _clean_fenced(raw)
        eval_result = json.loads(eval_text)
        sel = eval_result.get("selected", "").strip()
        reason = eval_result.get("reason", {})
        idx = int(sel.replace("후보", "").strip()) - 1
        return candidates[idx], sel, reason

    def _parse_json(self, best_output: str, candidates: List[str], selected: str) -> dict:
        """JSON 파싱"""
        try:
            return json.loads(best_output)
        except json.JSONDecodeError:
            idx = int(selected.replace("후보", "").strip()) - 1
            return json.loads(candidates[idx])

    def generate(
        self,
        input_data: Optional[dict] = None,
        mode: str = "test",
        source: str = "latest",
        rounds: Optional[int] = None
    ) -> Tuple[Dict, Dict[str, str], Dict, dict]:
        """
        메인 생성 함수 - 첫 번째 코드의 완전한 데이터 처리 + 두 번째 코드의 구조
        """
        # 입력 데이터 준비
        loaded_input, src_label = self._prepare_input(input_data, mode, source)
        
        if rounds is None:
            rounds = self.default_nway_rounds if input_data is not None else 2

        # 첫 번째 코드의 컨텍스트 빌딩 로직 적용
        ctx = _build_prompt_context(loaded_input)
        
        # 계획용 이미지 배치(ref)
        image_placements = _place_images_plan(loaded_input)

        # 프롬프트 포맷팅 (SafeDict 사용)
        prompt = self.gen_template.format_map(SafeDict(ctx))
        
        # 후보 생성
        raw_candidates = self._generate_candidates(prompt, rounds)
        # map to labels
        candidates = {f"후보 {i+1}": raw_candidates[i] for i in range(len(raw_candidates))}

        # 평가
        if rounds == 2:
            best_output, selected, reason = self.evaluate_candidates(raw_candidates)
        else:
            best_output, selected, reason = self.evaluate_candidates_nway(raw_candidates)

        # JSON 파싱 및 폴백 처리
        try:
            result = self._parse_json(best_output, raw_candidates, selected)
        except:
            result = {}
            
        # 첫 번째 코드의 폴백 로직 적용
        plan = result if isinstance(result, dict) and result else {}
        ok_sections = False
        try:
            sec = _safe(plan.get("content_plan"), dict, {}).get("sections", {})
            ok_sections = isinstance(sec, dict) and len(sec.keys()) >= 7
        except Exception:
            ok_sections = False
            
        if not plan or not ok_sections:
            plan = _fallback_plan(ctx, image_placements)
            result = plan

        # 병원 메타(노출 제한 필드만 포함)
        hospital_meta = {
            "name": ctx.get("hospital_name",""),
            "city": ctx.get("hospital_city","") or ctx.get("city",""),
            "district": ctx.get("hospital_district","") or ctx.get("district",""),
            "homepage": ctx.get("hospital_homepage","") or ctx.get("homepage",""),
            "map_link": ctx.get("hospital_map_link","") or ctx.get("map_link",""),
            "business_card": Path(ctx.get("hospital_business_card_path")).name if ctx.get("hospital_business_card_path") else "",
        }

        # 계획 린트(옵션)
        lint_report = _lint_plan(plan, enabled=self.lint_mode)

        # 최종 결과 구조 (첫 번째 코드 스타일)
        final_result = {
            "plan": plan,
            "section_titles": {sid: title for _, sid, title in SECTION_ORDER},
            "sections_order": [sid for _, sid, _ in SECTION_ORDER],
            "image_placements": image_placements,
            "hospital_meta": hospital_meta,
            "notes": {
                "ad_compliance": "본문/제목 생성 시 병원명/전화/가격/이벤트/내부링크 금지, 푸터에만 홈페이지/지도",
                "assets": "로고 미사용, 명함은 맨 아래1회"
            },
            "lint_report": lint_report,
            "source_input_path": src_label
        }
        
        evaluation_info = {"selected": selected, "reason": reason}
        return final_result, candidates, evaluation_info, loaded_input

    def save_log(
        self,
        input_data: dict,
        candidates: Dict[str, str],
        result: Dict[str, Any],
        selected: str,
        reason: dict,
        mode: str = "test",
        formatted_prompt: Optional[str] = None
    ) -> None:
        """로그 저장 - 첫 번째 코드의 상세 저장 방식 적용"""
        base = CLI_DIR if mode == "cli" else (TEST_DIR if mode == "test" else USE_DIR)
        base.mkdir(parents=True, exist_ok=True)
        ts = _now()

        # 최종 계획 → *_plan_log.json
        plan_log = {
            "plan": result.get("plan", {}),
            "section_titles": result.get("section_titles", {}),
            "sections_order": result.get("sections_order", []),
            "image_placements": result.get("image_placements", {}),
            "hospital_meta": result.get("hospital_meta", {}),
            "notes": result.get("notes", {}),
            "lint_report": result.get("lint_report", {}),
            "source_input_path": result.get("source_input_path", ""),
            "candidates": candidates,
            "selected": selected,
            "evaluation_reason": reason
        }
        
        with open(base / f"{ts}_plan_log.json", "w", encoding="utf-8") as f:
            json.dump(plan_log, f, ensure_ascii=False, indent=2)

        # 프롬프트 로그 → *_plan.json (첫 번째 코드 스타일)
        if formatted_prompt:
            with open(base / f"{ts}_plan.json", "w", encoding="utf-8") as f:
                json.dump({
                    "ts": ts,
                    "mode": f"{mode}-generate",
                    "input_data": input_data,
                    "generation_prompt": formatted_prompt,
                    "candidates": candidates,
                    "selected": selected,
                    "evaluation_reason": reason,
                    "notes": {
                        "sections": 7,
                        "image_placement_rule": "visit→2, therapy→5, result→7 기본 + 설명 키워드로 2~7 재배치",
                        "logo": "수신만, 출력 안 함",
                        "business_card": "맨 아래 1회 표시(있을 때만)",
                        "ad_compliance": "제목/본문은 별도 에이전트에서 생성 시 준수",
                        "lint_mode": self.lint_mode
                    }
                }, f, ensure_ascii=False, indent=2)

        print(f"✅ 저장(최종 계획): {base/(ts+'_plan_log.json')}")
        if formatted_prompt:
            print(f"📝 저장(로그):     {base/(ts+'_plan.json')}")

    def run(self, mode: str = "test", source: str = "latest") -> Dict[str, Any]:
        """실행 - 첫 번째 코드 스타일"""
        print(f"🧭 mode={mode}, source={source}")
        result, candidates, evaluation_info, loaded_input = self.generate(
            input_data=None, mode=mode, source=source
        )
        
        # 프롬프트 재생성 (저장용)
        ctx = _build_prompt_context(loaded_input)
        formatted_prompt = self.gen_template.format_map(SafeDict(ctx))
        
        self.save_log(
            input_data=loaded_input,
            candidates=candidates,
            result=result,
            selected=evaluation_info["selected"],
            reason=evaluation_info["reason"],
            mode=mode,
            formatted_prompt=formatted_prompt
        )
        return result

# =========================
# CLI 보조 매핑 (첫 번째 코드 스타일)
# =========================
def _resolve_mode(inp: str) -> str:
    v = (inp or "").strip().lower()
    if v in ("1", "test", "t"): return "test"
    if v in ("2", "use", "u"):  return "use"
    if v in ("3", "cli", "c"):  return "cli"
    return "test"

def _resolve_source(inp: str) -> str:
    v = (inp or "").strip().lower()
    if v in ("1", "latest", "l"):  return "latest"
    if v in ("2", "collect", "c"): return "collect"
    return "latest"

def _resolve_lint(inp: str) -> bool:
    v = (inp or "").strip().lower()
    if v in ("2", "on", "y", "yes"): return True
    return False

# =========================
# CLI
# =========================
if __name__ == "__main__":
    print("🔍 PlanAgent (통합 버전) 시작")
    print("모드 선택: 1) test  2) use  3) cli")
    mode = _resolve_mode(input("선택 (1/2/3): ").strip())

    print("입력 소스: 1) 최신 로그 사용  2) 지금 수집(InputAgent)")
    source = _resolve_source(input("선택 (1/2): ").strip())

    print("컴플라이언스 린터(계획 텍스트 검사): 1) OFF  2) ON")
    lint_mode = _resolve_lint(input("선택 (1/2): ").strip())

    agent = PlanAgent(
        gen_template_path=str(GEN_PROMPT_PATH),
        eval2_template_path=str(EVAL2_PROMPT_PATH),
        nway_template_path=str(NWAY_PROMPT_PATH),
        lint_mode=lint_mode
    )
    
    res = agent.run(mode=mode, source=source)
    
    print("\n=== PLAN PREVIEW ===")
    plan = res.get("plan", {})
    print("sections:", list(plan.get("content_plan", {}).get("sections", {}).keys()))
    print("assets:", plan.get("assets_plan", {}))
    print("hospital_meta:", res.get("hospital_meta", {}))