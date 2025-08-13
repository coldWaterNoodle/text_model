# agents/content_agent.py
# -*- coding: utf-8 -*-
"""
ContentAgent (7섹션 · 프롬프트 기반 · 이미지 바인딩 해석 · 의료광고 필터 · 로그 저장)
- 입력: 최신 input/plan/title 결과 자동 탐색 또는 경로 지정
- 프롬프트: test_prompt/content{1..7}_*.txt
- 모델: Gemini (GEMINI_API_KEY 필요)
- 저장:
    - 결과: test_logs/{mode}/{YYYYMMDD}/{timestamp}_content.json
    - 로그 : test_logs/{mode}/{YYYYMMDD}/{timestamp}_content_log.json
    - TXT : test_logs/{mode}/{YYYYMMDD}/{timestamp}_title_content_result.txt
"""

from __future__ import annotations

import os, re, json, time
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
import google.generativeai as genai

# =========================
# 환경설정 / 모델
# =========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")
genai.configure(api_key=API_KEY)

class GeminiClient:
    def __init__(self, model="models/gemini-1.5-flash", temperature=0.65, max_output_tokens=4096):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_retries = 3
        self.retry_delay = 1.0

    def generate(self, prompt: str, temperature: Optional[float] = None) -> str:
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
                    ),
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

gem = GeminiClient()

# =========================
# 유틸 (시간/경로/로딩)
# =========================
DEF_MODE = "use"

def _today() -> str: return datetime.now().strftime("%Y%m%d")
def _now() -> str:   return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _mtime(p: Path) -> float:
    try: return p.stat().st_mtime
    except Exception: return 0.0

def _read(path: Path, default=""):
    try: return path.read_text(encoding="utf-8")
    except Exception: return default

def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def _get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

# 최신 input 탐색 (신규/구형 모두)
def _latest_input(mode: str) -> Tuple[Optional[Path], Optional[dict]]:
    day = Path(f"test_logs/{mode}/{_today()}")
    patterns = ["*_input_logs.json", "*_input_log.json"]
    for pat in patterns:
        hits = sorted(day.glob(pat), key=_mtime, reverse=True)
        if hits:
            p = hits[0]
            try:
                data = _json_load(p)
                if isinstance(data, list) and data:
                    return p, data[-1]
                if isinstance(data, dict):
                    return p, data
            except Exception:
                pass
    root = Path(f"test_logs/{mode}")
    if not root.exists(): return None, None
    all_hits = sorted(list(root.rglob("*_input_logs.json")) + list(root.rglob("*_input_log.json")), key=_mtime, reverse=True)
    if not all_hits: return None, None
    p = all_hits[0]
    data = _json_load(p)
    if isinstance(data, list) and data: return p, data[-1]
    if isinstance(data, dict): return p, data
    return None, None

def _latest_plan(mode: str) -> Optional[Path]:
    day = Path(f"test_logs/{mode}/{_today()}")
    hits = sorted(day.glob("*_plan.json"), key=_mtime, reverse=True)
    if hits: return hits[0]
    root = Path(f"test_logs/{mode}")
    if not root.exists(): return None
    hits = sorted(root.rglob("*_plan.json"), key=_mtime, reverse=True)
    return hits[0] if hits else None

def _latest_title(mode: str) -> Optional[Path]:
    day = Path(f"test_logs/{mode}/{_today()}")
    hits = sorted(day.glob("*_title.json"), key=_mtime, reverse=True)
    if hits: return hits[0]
    root = Path(f"test_logs/{mode}")
    if not root.exists(): return None
    hits = sorted(root.rglob("*_title.json"), key=_mtime, reverse=True)
    return hits[0] if hits else None

# =========================
# 프롬프트 로딩/치환
# =========================
PROMPTS = {
    "1_intro":       Path("test_prompt/content1_intro_prompt.txt"),
    "2_visit":       Path("test_prompt/content2_visit_prompt.txt"),
    "3_inspection":  Path("test_prompt/content3_inspection_prompt.txt"),
    "4_doctor_tip":  Path("test_prompt/content4_doctor_tip_prompt.txt"),
    "5_treatment":   Path("test_prompt/content5_treatment_prompt.txt"),
    "6_check_point": Path("test_prompt/content6_check_point_prompt.txt"),
    "7_conclusion":  Path("test_prompt/content7_conclusion_prompt.txt"),
}

### 디버그용 프롬프트 로딩
print("프롬프트 로딩:", ", ".join(f"{k}={v.name}" for k, v in PROMPTS.items()))

def _render_template(tpl: str, ctx_vars: Dict[str, Any]) -> str:
    # 보호용: 이중 중괄호는 살림
    L, R = "§§L§§", "§§R§§"
    work = tpl.replace("{{", L).replace("}}", R)

    # {변수}만 안전 치환
    keys = list(ctx_vars.keys())
    if keys:
        pattern = re.compile(r"\{(" + "|".join(map(re.escape, keys)) + r")\}")
        work = pattern.sub(lambda m: str(ctx_vars.get(m.group(1), "")), work)

    return work.replace(L, "{{").replace(R, "}}")

# =========================
# JSON 파싱 & 텍스트 필터
# =========================
FORBIDDEN = [
    r"\b100%\b", r"무통증", r"완치", r"유일", r"최고", r"즉시\s*효과", r"파격", r"이벤트", r"특가",
    r"\d+\s*원", r"\d+\s*만원", r"가격\s*", r"전화\s*\d", r"http[s]?://", r"www\."
]
FORBIDDEN_RE = re.compile("|".join(FORBIDDEN))

def _clean_output(text: str) -> str:
    s = (text or "").strip()
    # 코드펜스 제거
    s = re.sub(r"^```(markdown|text)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # 과도한 공백 정리
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    # 금칙어 간단 마스킹(완전 삭제 대신 안전표기)
    s = FORBIDDEN_RE.sub(lambda _: "(광고성 문구 제거)", s)
    return s

def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s

# 동물 이미지 GIF
import random
GIF_DIR = Path("test_data/test_image/gif")

_EMOTICON_MARK_RE = re.compile(r"\((행복|슬픔|신남|화남|일반|마무리)\)")
# 게시글 단위로 동물 고정 & 풀 캐시
_SESSION: Dict[str, Any] = {"animal": None, "pool": None}

def _scan_gif_pool() -> Dict[str, Dict[str, List[Path]]]:
    """
    pool[animal][category] = [Path, ...]
    파일명 패턴 예: 행복_토끼.gif / 일반_햄스터3.gif / 마무리_토끼2.gif
    """
    pool: Dict[str, Dict[str, List[Path]]] = {}
    if not GIF_DIR.exists():
        return pool
    for p in GIF_DIR.glob("*.gif"):
        name = p.stem  # ex) 행복_토끼2
        parts = name.split("_", 1)
        if len(parts) < 2:
            continue
        category, animal_with_no = parts[0], parts[1]
        # 숫자 접미 제거
        animal = re.sub(r"\d+$", "", animal_with_no)
        animal = animal.strip()
        d = pool.setdefault(animal, {})
        d.setdefault(category, []).append(p)
    return pool

def _pick_animal_once(state: Dict[str, Any], pool: Dict[str, Dict[str, List[Path]]], preferred: Optional[str] = None) -> Optional[str]:
    if state.get("chosen_animal"):
        return state["chosen_animal"]
    candidates = list(pool.keys())
    if not candidates:
        return None
    if preferred and preferred in candidates:
        state["chosen_animal"] = preferred
        return preferred
    # 랜덤 고정
    animal = random.choice(candidates)
    state["chosen_animal"] = animal
    return animal

def _pick_gif_by(animal: str, category: str, pool: Dict[str, Dict[str, List[Path]]]) -> Optional[Path]:
    cand = (pool.get(animal, {}) or {}).get(category, [])
    if not cand:
        return None
    return random.choice(cand)

def _gif_pool_cached() -> Dict[str, Dict[str, List[Path]]]:
    if _SESSION["pool"] is None:
        _SESSION["pool"] = _scan_gif_pool()
    return _SESSION["pool"]

def _inject_emoticons_inline(text: str, sec_key: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    (행복/슬픔/놀람/신남/화남/일반/마무리) 마커를 같은 동물 GIF로 치환.
    - 섹션 1~6: 첫 마커에서 동물 랜덤 고정. 카테고리 없으면 '일반' 폴백 허용.
    - 섹션 7: (마무리)만 처리, '마무리_동물*' 없거나 동물 미고정이면 삽입하지 않음.
    """
    if not text:
        return text, []

    pool = _gif_pool_cached()
    images_log: List[Dict[str, str]] = []

    def repl(m: re.Match) -> str:
        tag = m.group(1)

        # 섹션7 전용 규칙
        if sec_key == "7_conclusion":
            if tag != "마무리":
                return ""  # 다른 마커는 제거
            animal = _SESSION.get("animal")
            if not animal:
                return ""  # 앞 섹션에서 동물 확정 안 됨 → 삽입 안 함
            media = _pick_gif_by(animal, "마무리", pool)
            if not media:
                return ""  # 마무리_동물 파일 없음 → 삽입 안 함
            alt = f"마무리 {animal} 이모티콘"
            images_log.append({"filename": media.name, "path": str(media), "alt": alt, "position": "inline"})
            return f"![{alt}]({str(media)})"

        # 섹션 1~6: 동물 없으면 지금 랜덤 고정
        animal = _SESSION.get("animal")
        if not animal:
            if not pool:
                return ""  # 풀 비어있으면 제거
            animal = random.choice(list(pool.keys()))
            _SESSION["animal"] = animal

        # 카테고리 선택: '마무리' 마커가 1~6에 오면 '일반'로 처리
        desired = "일반" if tag == "마무리" else tag
        media = _pick_gif_by(animal, desired, pool) or (None if desired == "일반" else _pick_gif_by(animal, "일반", pool))
        if not media:
            return ""  # 해당/일반 모두 없으면 제거

        alt = f"{desired} {animal} 이모티콘"
        images_log.append({"filename": media.name, "path": str(media), "alt": alt, "position": "inline"})
        return f"![{alt}]({str(media)})"

    new_text = _EMOTICON_MARK_RE.sub(repl, text)
    return new_text, images_log

# =========================
# [NEW] 전역 dedup/경로정규화/해시/페어링 유틸
# =========================
import hashlib  # [NEW]

def _norm_path(p: str) -> str:  # [NEW]
    p = (p or "").strip().replace("\\", "/")
    p = re.sub(r"[?#].*$", "", p)  # 쿼리/프래그먼트 제거
    return p.lower()

def _file_hash_safe(p: str) -> Optional[str]:  # [NEW]
    """
    동일 파일이 경로만 다른 복사본일 수 있어, 해시를 우선 키로 사용(선택).
    실패 시 None 반환하여 경로 기반으로 대체.
    """
    try:
        with open(p, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def _dedup_key_for_image(im: Dict[str, str]) -> str:  # [NEW]
    path = _norm_path(im.get("path", ""))
    h = _file_hash_safe(path)
    return f"hash:{h}" if h else f"path:{path}"

def _limit_for_section(sec_key: str) -> int:  # [NEW]
    return {
        "1_intro": 1,
        "2_visit": 2,
        "3_inspection": 2,
        "4_doctor_tip": 2,
        "5_treatment": 6,
        "6_check_point": 1,
        "7_conclusion": 6,
    }.get(sec_key, 2)

_BEFORE_RE = re.compile(r"(?:^|[\s_\-])(전|before)(?:$|[\s_\-])", re.I)   # [NEW]
_AFTER_RE  = re.compile(r"(?:^|[\s_\-])(후|after)(?:$|[\s_\-])", re.I)    # [NEW]

def _pair_before_after(images: List[Dict[str, str]]) -> List[Dict[str, str]]:  # [NEW]
    """
    Q7 전/후 페어링 정렬: 파일명/alt에서 전/후 단서를 찾아 '전→후' 순으로 근접 배치
    단순 휴리스틱: index 순서 유지하되, 전/후 후보를 분리 후 interleave
    """
    befores, afters, others = [], [], []
    for im in images:
        keyspace = f"{im.get('filename','')} {im.get('alt','')}"
        if _BEFORE_RE.search(keyspace):
            befores.append(im)
        elif _AFTER_RE.search(keyspace):
            afters.append(im)
        else:
            others.append(im)
    paired: List[Dict[str, str]] = []
    n = max(len(befores), len(afters))
    for i in range(n):
        if i < len(befores): paired.append(befores[i])
        if i < len(afters):  paired.append(afters[i])
    return paired + others

def _dedup_and_limit_images(section_key: str,
                            images: List[Dict[str, str]],
                            used_keys: set) -> List[Dict[str, str]]:  # [NEW]
    """
    - inline 이미지는 렌더 대상 아님 → 배열에서 제외(로그는 별개)
    - 전역 dedup(해시 우선, 실패 시 경로)
    - 섹션별 상한 적용
    - Q7은 전/후 페어링 정렬
    """
    # 0) inline 제외 (assemble에서 무시되지만 로그 혼입 방지)
    filtered = [im for im in images if (im.get("position") or "").lower() != "inline"]

    # 1) 전역 dedup
    unique: List[Dict[str, str]] = []
    for im in filtered:
        # path 우선 설정(입력 path가 있으면 그걸 사용)
        p = im.get("path") or ""
        if not p and im.get("filename"):
            p = f"test_data/test_image/{im['filename']}"
            im["path"] = p

        key = _dedup_key_for_image(im)
        if key in used_keys:
            continue
        used_keys.add(key)
        unique.append(im)

    # 2) Q7 페어링
    if section_key == "7_conclusion":
        unique = _pair_before_after(unique)

    # 3) 섹션별 상한
    limit = _limit_for_section(section_key)
    if len(unique) > limit:
        unique = unique[:limit]

    return unique

# =========================
# 이미지 바인딩 해석
# =========================
def _resolve_images_for_section(plan_sec: Dict[str, Any], input_row: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    확장 사항:
    - 배열 소스에 random 선택 지원: image_binding 항목에 "random": true
    - GIF 자동 선택 지원:
        * image_binding 항목에 {"from":"gif_pool", "category":"행복", "position":"bottom", "animal":"토끼"} 등
        * category 미지정 시 ["일반"] 시도, 섹션7(마무리)는 plan에서 category="마무리" 주길 권장
        * animal 미지정 시 글 단위로 랜덤 1종 고정
        * 여러 후보 카테고리를 시도하려면 "category_try": ["행복","일반"] 사용
    - 기존 동작(명함/hospital.business_card, question*_images 배열)은 그대로 유지
    """
    binds = plan_sec.get("image_binding") or []
    out: List[Dict[str, str]] = []

    for b in binds:
        src = b.get("from", "")
        limit = int(b.get("limit", 1))
        position = b.get("position", "top")

        # 1) GIF 풀에서 선택 (감정/일반/마무리 등)
        if src == "gif_pool":
            # 풀 캐시 확보
            pool = _SESSION.get("pool") or _scan_gif_pool()
            _SESSION["pool"] = pool

            # 동물 한 번 고정 (선호 동물이 오면 그걸 우선)
            preferred_animal = b.get("animal")  # 예: "토끼" / "햄스터" 등
            animal = _pick_animal_once(_SESSION, pool, preferred=preferred_animal)
            if animal:
                # 카테고리 후보: category_try > category > 기본 ["일반"]
                cat_try = b.get("category_try") or []
                if not cat_try:
                    cat = (b.get("category") or "").strip()
                    cat_try = [cat] if cat else ["일반"]

                picked = None
                for cat in cat_try:
                    picked = _pick_gif_by(animal, cat, pool)  # ← 함수명 교정
                    if picked:
                        break

                if picked:
                    out.append({
                        "filename": picked.name,
                        "path": str(picked),
                        "alt": f"{cat_try[0] if cat_try else '일반'} {animal} GIF",
                        "position": position
                    })
            continue

        # 2) 병원 명함 고정
        if src == "hospital.business_card":
            card = _get(input_row, "hospital.business_card", "")
            if card:
                out.append({
                    "filename": card,
                    "path": f"test_data/hospital_image/{card}" if Path(f"test_data/hospital_image/{card}").exists()
                            else f"test_data/test_image/{card}",
                    "alt": f"{_get(input_row,'hospital.name','')} 명함",
                    "position": "bottom",
                })
            continue

        # 3) 배열 소스 (visit/therapy/result 등)
        keys = [k.strip() for k in src.split("|") if k.strip()]
        arr = []
        for k in keys:
            val = _get(input_row, k, [])
            if isinstance(val, list) and val:
                arr = val
                break
        if not arr:
            continue

        # 랜덤 옵션: {"random": true}
        is_random = bool(b.get("random", False))
        chosen = (random.sample(arr, min(limit, len(arr))) if is_random else arr[:limit])

        for it in chosen:
            fn = it.get("filename", "")
            alt = it.get("description", "") or "임상 이미지"
            # [NEW] 입력에 path가 있으면 그대로 사용, 없으면 기존 규칙으로 구성
            path = (it.get("path") or f"test_data/test_image/{fn}")
            out.append({"filename": fn, "path": path, "alt": alt, "position": position})

    return out


# =========================
# 섹션 생성
# =========================
SECTION_TITLE_MAP = {
    "1_intro": "서론",
    "2_visit": "내원·방문",
    "3_inspection": "검사·진단",
    "4_doctor_tip": "의료진 팁",
    "5_treatment": "치료 과정",
    "6_check_point": "체크포인트",
    "7_conclusion": "마무리·결과",
}

def _build_ctx_vars(plan: Dict[str, Any], input_row: Dict[str, Any], title_obj: Dict[str, Any]) -> Dict[str, Any]:
    city = _get(input_row, "city", "")
    district = _get(input_row, "district", "")
    region_phrase = (_get(input_row, "region_phrase", "") or f"{city} {district}".strip()).strip()
    return {
        "title": _get(title_obj, "selected.title", ""),
        "hospital_name": _get(input_row, "hospital.name", ""),
        "save_name": _get(input_row, "hospital.save_name", ""),
        "city": city,
        "district": district,
        "region_phrase": region_phrase,
        "category": _get(input_row, "category", ""),
        "selected_symptom": _get(input_row, "selected_symptom", ""),
        "selected_procedure": _get(input_row, "selected_procedure", ""),
        "selected_treatment": _get(input_row, "selected_treatment", ""),
        "tooth_numbers": ", ".join(_get(input_row, "tooth_numbers", []) or []),
        "question1_concept": _get(input_row, "question1_concept", ""),
        "question2_condition": _get(input_row, "question2_condition", ""),
        "question4_treatment": _get(input_row, "question4_treatment", ""),
        "question6_result": _get(input_row, "question6_result", ""),
        "question8_extra": _get(input_row, "question8_extra", ""),
        "representative_persona": _get(input_row, "representative_persona", ""),
        "map_link": _get(input_row, "hospital.map_link", ""),
    }

def _build_section_prompt(sec_key: str, sec_plan: Dict[str, Any], base_ctx: Dict[str, Any]) -> str:
    # 외부 프롬프트 + 컨텍스트(JSON) + 섹션 가이드(요약/금지/필수)
    p_path = PROMPTS.get(sec_key)
    prompt_txt = _read(p_path, default=f"[{sec_key}]에 대한 본문을 한국어로 작성하세요.")
    prompt_txt = _render_template(prompt_txt, base_ctx)

    guide = {
        "section_key": sec_key,
        "section_title": SECTION_TITLE_MAP.get(sec_key, sec_key),
        "summary": sec_plan.get("summary", ""),
        "must_include": sec_plan.get("must_include", []),
        "may_include": sec_plan.get("may_include", []),
        "must_not_include": sec_plan.get("must_not_include", []),
        "style_rules": [
            "의료광고법 위반 표현 금지(가격/이벤트/과장/단정/유일/무통증/완치 등).",
            "정보제공 목적의 중립적 톤. 개인차/주의 유의미 암시.",
            "같은 문장·메시지 반복 금지, 문장 길이·줄바꿈은 자연스럽게.",
        ],
        "format_rules": [
            "불필요한 헤딩/번호 매기기 금지(프롬프트가 요구한 경우 제외).",
            "이모지는 프롬프트가 요구한 경우에만 제한적으로 사용.",
        ],
    }
    sys_dir = (
      "You are a Korean medical blog writer. Follow all rules. "
      "Return PLAIN TEXT only (no JSON, no backticks)."
    )
    final = f"{sys_dir}\n\nINSTRUCTION\n{prompt_txt}\n\nCONTEXT(JSON)\n{json.dumps(base_ctx, ensure_ascii=False, indent=2)}\n\nSECTION_GUIDE(JSON)\n{json.dumps(guide, ensure_ascii=False, indent=2)}\n\nWrite the section now:"
    return final

# =========================
# 본문 조립
# =========================
def _assemble_markdown(sections_out: Dict[str, Dict[str, Any]]) -> str:
    parts: List[str] = []
    for k in ["1_intro","2_visit","3_inspection","4_doctor_tip","5_treatment","6_check_point","7_conclusion"]:
        sec = sections_out.get(k)
        if not sec: continue
        imgs_top = [im for im in sec.get("images", []) if im.get("position") == "top"]
        for im in imgs_top:
            parts.append(f"![{im.get('alt','')}]({im.get('path','')})")
        parts.append(sec.get("text", "").rstrip())
        imgs_bottom = [im for im in sec.get("images", []) if im.get("position") == "bottom"]
        for im in imgs_bottom:
            parts.append(f"![{im.get('alt','')}]({im.get('path','')})")
        parts.append("")  # 섹션 사이 공백줄
    return "\n".join(parts).strip()

# ===== 복붙용 변환 =====
_IMG_MD_RE = re.compile(r'!\[([^\]]*)\]\(([^)]+)\)')

def _to_title_content_result(title: str, md: str) -> str:
    """
    - 첫 줄에 제목
    - 공백 줄 1개
    - 본문에서 ![ALT](PATH)를
        (PATH)
        [ALT]
      로 변환
    """
    body = md or ""
    def _img_repl(m: re.Match):
        alt = (m.group(1) or "").strip()
        path = (m.group(2) or "").strip()
        pnorm = path.lower().replace("\\", "/")  # 경로 정규화
        if pnorm.endswith(".gif") or "/test_data/test_image/gif/" in pnorm:
            return f"\n({path})\n"
        return f"\n({path})\n[{alt}]\n"

    body = _IMG_MD_RE.sub(_img_repl, body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    title_line = (title or "").strip()
    if title_line:
        return f"{title_line}\n\n{body}".strip()
    return body

# =========================
# 저장
# =========================
def _save_json(mode: str, name: str, payload: dict) -> Path:
    out_dir = Path(f"test_logs/{mode}/{_today()}")
    _ensure_dir(out_dir)
    p = out_dir / f"{_now()}_{name}.json"
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

# =========================
# 실행
# =========================
def run(mode: str = DEF_MODE,
        input_path: Optional[str|Path] = None,
        plan_path: Optional[str|Path] = None,
        title_path: Optional[str|Path] = None) -> Dict[str, Any]:

    # 1) 입력 수집
    if input_path:
        inp_path = Path(input_path); inp_row = _json_load(inp_path)
        if isinstance(inp_row, list) and inp_row: inp_row = inp_row[-1]
        inp_src = str(inp_path)
    else:
        found_path, row = _latest_input(mode)
        if row is None:
            raise FileNotFoundError("최신 *_input_log(s).json을 찾지 못했습니다. 먼저 InputAgent를 실행하세요.")
        inp_row, inp_src = row, str(found_path)

    if plan_path:
        plan = _json_load(Path(plan_path)); plan_src = plan_path
    else:
        p = _latest_plan(mode)
        if not p: raise FileNotFoundError("최신 *_plan.json을 찾지 못했습니다. 먼저 PlanAgent를 실행하세요.")
        plan = _json_load(p); plan_src = str(p)

    if title_path:
        title_obj = _json_load(Path(title_path)); title_src = title_path
    else:
        t = _latest_title(mode)
        if not t: raise FileNotFoundError("최신 *_title.json을 찾지 못했습니다. 먼저 TitleAgent를 실행하세요.")
        title_obj = _json_load(t); title_src = str(t)

    # 2) 컨텍스트 준비
    base_ctx = _build_ctx_vars(plan, inp_row, title_obj)
    order = _get(plan, "content_plan.sections_order", []) or ["1_intro","2_visit","3_inspection","4_doctor_tip","5_treatment","6_check_point","7_conclusion"]
    sections_plan: Dict[str, Any] = _get(plan, "content_plan.sections", {}) or {}

    # 3) 섹션별 생성
    sections_out: Dict[str, Dict[str, Any]] = {}
    log_detail: Dict[str, Any] = {"sections": {}}

    used_image_keys: set = set()  # [NEW] 전역 dedup 키 저장소

    for k in order:
        sec_plan = sections_plan.get(k, {})
        prompt = _build_section_prompt(k, sec_plan, base_ctx)
        raw = gem.generate(prompt)
        text = _clean_output(raw)

        # ✅ 이모티콘 마커 치환을 섹션별로 적용
        text, emoticon_imgs = _inject_emoticons_inline(text, k)

        # 후보 이미지 수집
        images = _resolve_images_for_section(sec_plan, inp_row)

        # 로그용 inline도 합치되, 렌더 중복 방지를 위해 dedup 단계에서 inline 제거
        if emoticon_imgs:
            images.extend(emoticon_imgs)

        # [NEW] 전역 dedup + 섹션 상한 + Q7 전/후 페어링
        images = _dedup_and_limit_images(k, images, used_image_keys)

        sections_out[k] = {
            "title": SECTION_TITLE_MAP.get(k, k),
            "text": text,
            "images": images
        }
        log_detail["sections"][k] = {
            "prompt_path": str(PROMPTS.get(k, "")),
            "prompt_rendered_preview": prompt[:1200],
            "llm_raw_preview": raw[:1200],
            "used_summary": sec_plan.get("summary", ""),
            "resolved_images": images,
        }

    # 4) 최종 조립 → 복붙용 문자열 생성
    md = _assemble_markdown(sections_out)
    title_content_result = _to_title_content_result(base_ctx.get("title", ""), md)

    # 5) 저장 (assembled_markdown에 복붙용 문자열을 저장하고, title_content_result 필드는 제거)
    result = {
        "meta": {
            "mode": mode,
            "timestamp": _now(),
            "model": gem.model,
            "temperature": gem.temperature,
            "max_output_tokens": gem.max_output_tokens,
            "plan_source": plan_src,
            "input_source": inp_src,
            "title_source": title_src,
            "case_id": _get(inp_row, "case_id", ""),
        },
        "title": base_ctx.get("title", ""),
        "sections": sections_out,
        "assembled_markdown": title_content_result,  # ✅ 복붙용 문자열로 교체
    }
    out_path = _save_json(mode, "content", result)

    # 동일 내용 TXT 저장
    out_dir = out_path.parent
    ts_prefix = out_path.stem.replace("_content", "")  # 예: 20250812_141055
    txt_path = out_dir / f"{ts_prefix}_title_content_result.txt"
    txt_path.write_text(title_content_result, encoding="utf-8")

    # 로그
    log = {
        "meta": {
            "mode": mode,
            "timestamp": _now(),
            "success": True,
        },
        "context_vars": base_ctx,
        "output_paths": {
            "content_path": str(out_path),
            "title_content_txt": str(txt_path),
        },
        **log_detail,
    }
    log_path = _save_json(mode, "content_log", log)

    print(f"✅ Content 저장: {out_path}")
    print(f"🧾 로그 저장: {log_path}")
    print(f"📝 복붙용 TXT 저장: {txt_path}")
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ContentAgent — plan/title/input 기반 7섹션 본문 생성")
    ap.add_argument("--mode", default=DEF_MODE, choices=["test","use"])
    ap.add_argument("--input", default="", help="*_input_log(s).json 경로(미지정 시 최신)")
    ap.add_argument("--plan",  default="", help="*_plan.json 경로(미지정 시 최신)")
    ap.add_argument("--title", default="", help="*_title.json 경로(미지정 시 최신)")
    args = ap.parse_args()

    run(mode=args.mode,
        input_path=(args.input or None),
        plan_path=(args.plan or None),
        title_path=(args.title or None))
