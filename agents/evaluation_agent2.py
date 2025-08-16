# -*- coding: utf-8 -*-
"""
EvaluationAgent (Router + Threshold Fix + Plan/Title/Content Agent routing)

핵심 기능
- 최신 파일 탐색: 패턴 '순서' 우선 → (루트→최신날짜폴더→재귀)
- Title/Plan 산출물 안전 추출 및 합성
- SEO/의료법 임계치 해석 분리 (SEO=min, 의료법=max) + {min,max} 지원
- Router 모드: 문제 유형별로 Plan/Title/Content 에이전트 호출 → 산출물 재로딩 → 재평가
- 비Router 모드: 기존 LLM 국소수정(fallback) 루프 유지
- 가중 총점 계산 시점 오류(UnboundLocal) 제거: 매 사용 직전 즉시 계산
- SEO gs()의 이미지(9번) 점수 규칙 간단 정리

필수: .env에 GEMINI_API_KEY
"""

import os
import re
import csv
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union, Iterable, Optional

import google.generativeai as genai
from dotenv import load_dotenv

# ===== 경로 기본 =====
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = ROOT / "test_logs" / "test"
PROMPTS_DIR = ROOT / "test_prompt"
DATA_DIR = ROOT / "test_data"

EVAL_PROMPT_PATH = PROMPTS_DIR / "llm_evaluation_prompt.txt"
REGEN_PROMPT_PATH = PROMPTS_DIR / "llm_regeneration_prompt.txt"
SEO_PROMPT_PATH = PROMPTS_DIR / "seo_evaluation_prompt.txt"

CRITERIA_PATH = DATA_DIR / "medical_evaluation_criteria.json"
SEO_CRITERIA_PATH = DATA_DIR / "seo_evaluation_criteria.json"

DEFAULT_CSV_PATHS = [DATA_DIR / "medical_ad_checklist.csv", Path("/mnt/test_data/medical_ad_checklist.csv")]
DEFAULT_REPORT_PATHS = [DATA_DIR / "medical-ad-report.md", Path("/mnt/test_data/medical-ad-report.md")]

# ===== 체크리스트 명칭 =====
CHECKLIST_NAMES = {
    1: "허위·과장 표현", 2: "치료경험담", 3: "비급여 진료비 할인", 4: "사전심의 미이행",
    5: "치료 전후 사진", 6: "전문의 허위 표시", 7: "환자 유인·알선", 8: "비의료인 의료광고",
    9: "객관적 근거 부족", 10: "비교 광고", 11: "기사형 광고", 12: "부작용 정보 누락",
    13: "인증·보증 허위표시", 14: "가격 정보 오표시", 15: "연락처 정보 오류",
}

SEO_CHECKLIST_NAMES = {
    1: "제목 글자수 (공백 포함)", 2: "제목 글자수 (공백 제외)", 3: "본문 글자수 (공백 포함)",
    4: "본문 글자수 (공백 제외)", 5: "총 형태소 개수", 6: "총 음절 개수",
    7: "총 단어 개수", 8: "어뷰징 단어 개수", 9: "본문 이미지"
}

# ===== 리포트 가중치 (기본값) =====
DEFAULT_REPORT_WEIGHTS = {
    "1": 8.6, "2": 8.0, "3": 8.0, "4": 8.0, "5": 7.0,
    "6": 7.0, "7": 8.0, "8": 7.4, "9": 6.4, "10": 6.4,
    "11": 6.0, "12": 6.0, "13": 6.0, "14": 6.0, "15": 5.5
}

# ===== 규칙 엔진 기본 패턴 =====
BASE_PATTERNS = {
    1: [r"\b100\s*%\b", r"부작용\s*없(음|다)", r"\b최고\b", r"\b유일(한)?\b", r"완전\s*무통"],
    2: [r"후기|경험담|리뷰", r"만족도", r"치료\s*과정", r"치료\s*결과", r"협찬|제공\s*받"],
    3: [r"\d{1,3}\s?%(\s*할인)?", r"이벤트\s*가", r"행사\s*가", r"\b원\s*부터\b"],
    4: [r"심의번호", r"심의\s*미이행|미심의"],
    5: [r"\b전후\b", r"\bbefore\b", r"\bafter\b", r"!\[.*\]\(.*\)", r"<img[^>]+>"],
    6: [r"전문의", r"전문병원", r"임플란트\s*전문의", r"교정\s*전문병원"],
    7: [r"리뷰\s*이벤트", r"추첨", r"사은품", r"리뷰\s*작성\s*시", r"대가|포인트|기프티콘"],
    8: [r"인플루언서|일반인\s*광고", r"제휴\s*포스팅"],
    9: [r"임상결과|연구결과|데이터", r"근거\s*없(음|다)"],
    10:[r"타\s*병원|다른\s*병원", r"최초|최고|유일\s*비교", r"보다\s*낫"],
    11:[r"기사형|보도자료|인터뷰\s*형태", r"전문가\s*의견\s*형식"],
    12:[r"부작용|주의사항|개인차", r"리스크|합병증"],
    13:[r"인증|상장|감사장|추천", r"공식\s*인증"],
    14:[r"원\s*부터|최저가|할인\s*가", r"추가\s*비용|부가세"],
    15:[r"병원명|주소|전화|연락처", r"오류|불일치"],
}

# ===== SEO 메트릭 =====
def _calculate_morphemes(text: str) -> int:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    result = kiwi.analyze(text)
    tokens, _ = result[0]
    return len([t.form for t in tokens])

def _count_syllables_extended(text: str) -> int:
    import unicodedata
    text = unicodedata.normalize('NFC', text)
    return sum(1 for ch in text if (0xAC00 <= ord(ch) <= 0xD7A3) or (ch.isascii() and ch.isalpha()))

def calculate_seo_metrics(title: str, content: str) -> Dict[str, int]:
    import re
    return {
        1: len(title),
        2: len(title.replace(" ", "")),
        3: len(content),
        4: len(content.replace(" ", "")),
        5: _calculate_morphemes(content),
        6: _count_syllables_extended(content),
        7: len(re.findall(r'[\w가-힣]+', content)),
        8: sum(len(re.findall(p, content, re.IGNORECASE)) for p in [
            r'19금', r'성인', r'유해', r'도박', r'불법', r'사기',
            r'100%', r'완전무료', r'대박', r'짱', r'헐', r'1등', r'최고', r'최강', r'완벽', r'보장', r'완치', r'치료보장',
            r'즉시', r'당일', r'바로', r'지금\s*당장', r'반드시', r'절대', r'무조건',
            r'전부', r'전세계', r'국내유일', r'독점', r'유일무이', r'베스트', r'프리미엄',
            r'명품', r'초특가', r'파격', r'무료', r'공짜', r'할인', r'이벤트', r'사은품',
            r'한정', r'마감임박', r'재고소진', r'선착순', r'단독', r'최초', r'유일',
            r'완전', r'필수', r'강력추천'
        ]),
        9: len(re.findall(r'!\[.*?\]\(.*?\)', content)) + len(re.findall(r'<img[^>]*>', content))
    }

# ===== 유틸 =====
def _nowstamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _today() -> str:
    return datetime.now().strftime("%Y%m%d")

def _read_text(p: Path) -> str:
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {p}")
    return p.read_text(encoding="utf-8")

def _read_json(p: Path) -> Any:
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {p}")
    return json.loads(p.read_text(encoding="utf-8"))

def _write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

# ---- 임계치 정규화: SEO=min, 의료법=max ----
def _normalize_thresholds(th_map: Dict[str, Any], evaluation_mode: str) -> Dict[str, Dict[str, Optional[int]]]:
    """
    숫자 하나면 medical=max, seo=min로 해석. dict면 {min,max} 그대로 사용.
    """
    out: Dict[str, Dict[str, Optional[int]]] = {}
    for k, v in th_map.items():
        if isinstance(v, dict):
            mn = v.get("min", None)
            mx = v.get("max", None)
        else:
            if evaluation_mode == "seo":
                mn, mx = int(v), None
            else:
                mn, mx = None, int(v)
        out[str(k)] = {"min": (int(mn) if mn is not None else None),
                       "max": (int(mx) if mx is not None else None)}
    return out

# ===== 파일 탐색(패턴 우선) =====
_DATE_DIR_RE = re.compile(r"^\d{8}$")  # YYYYMMDD

def _latest_date_dir(root: Path) -> Path:
    if not root.exists(): return root
    ds = [p for p in root.iterdir() if p.is_dir() and _DATE_DIR_RE.match(p.name)]
    return max(ds, key=lambda p: p.name) if ds else root

def _glob_many(root: Path, patterns: List[str], recursive: bool = False) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(root.glob(f"**/{pat}" if recursive else pat))
    uniq, seen = [], set()
    for f in out:
        try:
            rp = f.resolve()
        except Exception:
            continue
        if rp not in seen and rp.exists() and rp.is_file():
            seen.add(rp); uniq.append(rp)
    return uniq

def _pick_latest(paths: List[Path]) -> Optional[Path]:
    return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

def _list_names(p: Path) -> List[str]:
    try:
        return sorted([c.name for c in p.iterdir()])
    except Exception:
        return []

def _latest(log_dir: Path, glob_pat: Union[str, List[str]]) -> Path:
    patterns = glob_pat if isinstance(glob_pat, list) else [glob_pat]
    for pat in patterns:
        hits = _glob_many(log_dir, [pat], recursive=False)
        if not hits:
            latest_day = _latest_date_dir(log_dir)
            if latest_day != log_dir:
                hits = _glob_many(latest_day, [pat], recursive=False)
        if not hits:
            hits = _glob_many(log_dir, [pat], recursive=True)
        if hits:
            return _pick_latest(hits)
    latest_day = _latest_date_dir(log_dir)
    raise FileNotFoundError(
        f"최신 파일을 찾을 수 없습니다: {log_dir}/{patterns}\n"
        f"현재 루트 목록: {_list_names(log_dir) or '(비어 있음)'}\n"
        f"최신 날짜 폴더: {latest_day.name if latest_day != log_dir else '(없음)'} 목록: {_list_names(latest_day) or '(비어 있음)'}"
    )

# ===== LLM =====
def _setup_llm():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY가 .env에 없습니다.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-1.5-pro")

def _extract_json(raw: str) -> Dict[str, Any]:
    if not raw:
        raise ValueError("LLM 응답이 비어 있습니다.")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{"); end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]
    return json.loads(text.strip().strip("`").strip())

def _call_llm(model, prompt: str) -> Dict[str, Any]:
    resp = model.generate_content(prompt)
    text = getattr(resp, "text", "") or ""
    if not text:
        try:
            cand0 = resp.candidates[0]
            parts = getattr(getattr(cand0, "content", None), "parts", []) or []
            text = "".join(getattr(p, "text", "") for p in parts if getattr(p, "text", ""))
        except Exception:
            pass
    if not text:
        raise RuntimeError("LLM 응답 파싱 실패(빈 응답). 프롬프트 또는 안전필터 확인.")
    return _extract_json(text)

# ===== 재귀 탐색 도구 =====
def _iter_paths(obj: Any, prefix: Tuple=()) -> Iterable[Tuple[Tuple, Any]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_paths(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _iter_paths(v, prefix + (i,))
    else:
        yield (prefix, obj)

def _path_to_str(path: Tuple) -> str:
    return ".".join(str(p) for p in path)

# --- plan 7섹션 본문 합성 ---
def _plan_to_text(plan_obj: dict) -> str:
    try:
        cp = plan_obj.get("content_plan", {})
        order = cp.get("sections_order") or []
        secs  = cp.get("sections") or {}
        blocks = []
        for key in order:
            sec = secs.get(key) or {}
            sub = (sec.get("subtitle") or "").strip()
            summ = (sec.get("summary") or "").strip()
            if sub or summ:
                hdr = f"## {sub}" if sub else ""
                block = "\n".join(x for x in [hdr, summ] if x)
                if block.strip():
                    blocks.append(block.strip())
        return "\n\n".join(blocks)
    except Exception:
        return ""

def _fallback_title_from_plan(plan_obj: dict) -> str:
    cv = (plan_obj.get("context_vars") or {})
    city = (cv.get("city","") or "").strip()
    district = (cv.get("district","") or "").strip()
    category = (cv.get("category","") or "").strip()
    region = (cv.get("region_phrase") or f"{city} {district}").strip()
    cand = " · ".join([x for x in [region, category, "치료 케이스 가이드"] if x])
    return cand or "진료 케이스 가이드"

TITLE_KEY_HINTS = ["title","post_title","page_title","doc_title","headline","h1"]
CONTENT_KEY_HINTS = [
    "content","body","post_content","article","markdown","md","html","text",
    "paragraph","paragraphs","section","sections","blocks","document","value",
    "assembled_markdown","title_content_result",
    "content_plan","summary","subtitle",
    "stitched","stitched_content","content_markdown","stitched.content_markdown"
]

def _score_title_candidate(s: str) -> float:
    if not isinstance(s, str): return -1
    l = len(s.strip())
    if l < 3: return -1
    score = 0.0
    if 10 <= l <= 120: score += 2.0
    elif l <= 200: score += 1.0
    if sum(ch in "#*{}[]" for ch in s) > 5: score -= 0.5
    return score + min(l/200.0, 1.0)

def _normalize_block_to_text(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts = []
        for it in val:
            if isinstance(it, str):
                parts.append(it)
            elif isinstance(it, dict):
                for k in ["description","text","content","paragraph","markdown","md","html","value","assembled_markdown","title_content_result","summary","subtitle"]:
                    v = it.get(k)
                    if isinstance(v, str) and v.strip():
                        parts.append(v.strip())
                kp = it.get("key_points")
                if isinstance(kp, list) and kp:
                    bullet = "\n".join(f"- {str(x).strip()}" for x in kp if str(x).strip())
                    if bullet:
                        parts.append(bullet)
        return "\n\n".join(p for p in parts if p.strip())
    if isinstance(val, dict):
        if "content_plan" in val and isinstance(val.get("content_plan"), dict):
            text = _plan_to_text(val)
            if text.strip():
                return text
        for k in ["description","assembled_markdown","title_content_result","markdown","md","html","text","content","body","value","summary"]:
            v = val.get(k)
            if isinstance(v, str) and v.strip():
                return v
        for k in ["paragraphs","sections","blocks"]:
            v = val.get(k)
            if isinstance(v, list) and v:
                s = _normalize_block_to_text(v)
                if s.strip(): return s
    return ""

def _extract_title_content(clog: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    cand_titles: List[Tuple[str, str, float]] = []
    cand_contents: List[Tuple[str, str, int]] = []

    # TitleAgent 산출물/로그 대응
    if isinstance(clog.get("selected"), dict):
        sel_title = clog["selected"].get("title")
        if isinstance(sel_title, str) and sel_title.strip():
            cand_titles.append(("selected.title", sel_title.strip(), _score_title_candidate(sel_title)))
    if isinstance(clog.get("plan_snapshot"), dict):
        ps_text = _plan_to_text(clog["plan_snapshot"])
        if ps_text.strip():
            cand_contents.append(("plan_snapshot.content_plan", ps_text, len(ps_text)))
        if not cand_titles:
            fb = _fallback_title_from_plan(clog["plan_snapshot"])
            if fb:
                cand_titles.append(("fallback.title_from_plan_snapshot", fb, _score_title_candidate(fb)))

    # stitched 계열
    try:
        stitched = clog.get("stitched") if isinstance(clog, dict) else None
        if isinstance(stitched, dict):
            sc = stitched.get("content_markdown")
            if isinstance(sc, str) and sc.strip():
                cand_contents.append(("stitched.content_markdown", sc.strip(), len(sc.strip())))
    except Exception:
        pass

    # 최상위 일반 키들
    for tk in ["title","post_title","page_title","doc_title","headline","h1"]:
        if isinstance(clog.get(tk), str) and clog[tk].strip():
            cand_titles.append((tk, clog[tk].strip(), _score_title_candidate(clog[tk])))
    for ck in ["assembled_markdown","title_content_result","content","body","markdown","md","html","text","summary"]:
        if isinstance(clog.get(ck), str) and clog[ck].strip():
            text = clog[ck].strip()
            cand_contents.append((ck, text, len(text)))

    # 재귀 탐색
    for path, val in _iter_paths(clog):
        pstr = _path_to_str(path)
        key_lower = str(path[-1]).lower() if path else ""
        if any(h in key_lower for h in TITLE_KEY_HINTS) and isinstance(val, str):
            s = val.strip()
            if s: cand_titles.append((pstr, s, _score_title_candidate(s)))
        if any(h in key_lower for h in CONTENT_KEY_HINTS):
            text = _normalize_block_to_text(val)
            if not text and isinstance(val, str):
                text = val
            if isinstance(text, str) and text.strip():
                cand_contents.append((pstr, text, len(text)))

    # 선택
    title, title_path = "", ""
    if cand_titles:
        cand_titles.sort(key=lambda x: x[2], reverse=True)
        title_path, title, _ = cand_titles[0]
    content, content_path = "", ""
    if cand_contents:
        cand_contents.sort(key=lambda x: (x[2] >= 300, x[2]), reverse=True)
        content_path, content, _ = cand_contents[0]

    # plan 폴백
    if not title and isinstance(clog, dict) and ("content_plan" in clog or "title_plan" in clog):
        title = _fallback_title_from_plan(clog)
        title_path = "fallback.title_from_plan"
    if not content and isinstance(clog, dict) and "content_plan" in clog:
        content = _plan_to_text(clog)
        content_path = "content_plan.sections(summary)"

    dbg = {
        "title_path": title_path,
        "content_path": content_path,
        "title_candidates": [{"path":p,"len":len(v),"score":sc} for (p,v,sc) in cand_titles[:5]],
        "content_candidates": [{"path":p,"len":l} for (p,_,l) in cand_contents[:5]],
    }
    return title.strip(), content.strip(), dbg

# ===== CSV/규칙 =====
def _find_existing(paths: List[Path]) -> Path:
    for p in paths:
        if p.exists(): return p
    raise FileNotFoundError(f"경로들 중 파일이 없습니다: {paths}")

def load_checklist_csv(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): (v.strip() if isinstance(v,str) else v) for k,v in r.items()})
    return rows

def compile_patterns(rows: List[Dict[str,str]]) -> Dict[int, List[re.Pattern]]:
    patterns: Dict[int, List[re.Pattern]] = {}
    for r in rows:
        try:
            idx = int(r.get("번호") or r.get("no") or r.get("index"))
        except Exception:
            continue
        p_list = BASE_PATTERNS.get(idx, []).copy()
        eval_method = (r.get("평가방법") or "").replace("<br>", "\n")
        for kw in ["최고","유일","완전","100%","부작용 없음","이벤트","할인","전후","before","after",
                   "리뷰","후기","협찬","가격","원부터","심의번호","전문의","전문병원","주의사항","부작용","개인차",
                   "인증","상장","감사장","추천","기사형","보도자료","인터뷰","타 병원","최초","유일"]:
            if kw in eval_method and kw not in p_list:
                p_list.append(re.escape(kw))
        try:
            patterns[idx] = [re.compile(p, flags=re.I) for p in p_list]
        except re.error:
            patterns[idx] = [re.compile(re.escape(p), flags=re.I) for p in p_list if p]
    return patterns

def rule_score_item(idx: int, text: str, pats: Dict[int, List[re.Pattern]]) -> Tuple[int, List[str]]:
    if idx not in pats or not pats[idx]:
        return 0, []
    hits = []
    for rgx in pats[idx]:
        m = rgx.search(text)
        if m: hits.append(m.group(0))
    if not hits: return 0, []
    strong = any(re.search(r"100\s*%|부작용\s*없", h, re.I) for h in hits)
    if idx == 1 and strong: return 5, hits
    if idx in [3,5,7,14] and len(hits) >= 2: return 5, hits
    return (2 if len(hits) == 1 else 3), hits

def rule_score_all(title: str, content: str, pats: Dict[int, List[re.Pattern]]) -> Dict[str, Dict[str, Any]]:
    text = f"{title}\n\n{content}"
    return {str(i): {"score": s, "hits": hits} for i, (s, hits) in
            ((i, rule_score_item(i, text, pats)) for i in range(1, 16))}

# ===== 가중 총점 =====
def parse_report_weights(md_path: Path) -> Dict[str, float]:
    try:
        md = _read_text(md_path)
        lines = md.splitlines()
        weights = {}
        in_table = False
        for ln in lines:
            if "| 순위 |" in ln and "우선순위 점수" in ln:
                in_table = True
                continue
            if in_table:
                if ln.strip().startswith("|------"): continue
                if not ln.strip().startswith("|"): break
                cells = [c.strip() for c in ln.strip().strip("|").split("|")]
                if len(cells) >= 3:
                    name = cells[1]; w_str = cells[2]
                    idx = next((k for k,v in CHECKLIST_NAMES.items() if v in name), None)
                    if idx:
                        try: weights[str(idx)] = float(w_str)
                        except: pass
        return weights if weights else DEFAULT_REPORT_WEIGHTS
    except Exception:
        return DEFAULT_REPORT_WEIGHTS

def weighted_total(final_scores: Dict[str,int], weights: Dict[str,float], evaluation_mode: str = "medical") -> float:
    if evaluation_mode == "seo":
        # SEO는 항목 합산(임계 대비 여부는 별도 판단)
        return round(sum(final_scores.get(str(i), 0) for i in range(1, 10)), 1)
    num = sum((final_scores.get(k,0)/5.0) * weights[k] for k in weights)
    den = sum(weights.values())
    return round((num/den)*100, 1) if den else 0.0

# ===== criteria 모드 자동 매핑 =====
def _resolve_criteria(criteria: Dict[str, Any], mode: str) -> Tuple[Dict[str, int], str]:
    if not isinstance(criteria, dict):
        raise ValueError("criteria 형식이 잘못되었습니다(최상위가 dict 아님).")

    req = (mode or "").strip()
    req_lower = req.lower()

    group_alias = {
        "엄격":"strict","strict":"strict","우수":"strict","최상":"strict","강화":"strict","높음":"strict",
        "표준":"standard","standard":"standard","양호":"standard","기본":"standard","중간":"standard",
        "유연":"lenient","lenient":"lenient","보통":"lenient","완화":"lenient","낮음":"lenient",
    }
    group = group_alias.get(req_lower, None)

    by_group_priority = {
        "strict":   ["엄격","strict","우수","최상","강화","높음"],
        "standard": ["표준","standard","양호","기본","중간"],
        "lenient":  ["유연","lenient","보통","완화","낮음"],
    }

    def collect_dicts(c: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = [c]
        if isinstance(c.get("modes"), dict):
            out.append(c["modes"])
        return out

    dicts_to_check = collect_dicts(criteria)

    def find_in_dicts(names: List[str]) -> Optional[Tuple[Dict[str,int], str]]:
        for d in dicts_to_check:
            if not isinstance(d, dict):
                continue
        # 이름 일치 탐색
            for name in names:
                for k, v in d.items():
                    if str(k).lower() == str(name).lower() and isinstance(v, dict):
                        return v, k
        return None

    # 요청 라벨 & 동의어
    search_names = [req] if req else []
    if group:
        search_names += [n for n in by_group_priority[group] if n.lower() != req_lower]
    found = find_in_dicts(search_names)
    if found:
        return found

    # 숫자 키만 있는 직접 임계 맵
    if all(isinstance(k, str) and k.isdigit() for k in criteria.keys()):
        return criteria, "direct"

    # standard 계열 폴백
    found = find_in_dicts(by_group_priority["standard"])
    if found:
        return found

    # 최후 폴백: 첫 dict 값
    for d in dicts_to_check:
        for k, v in d.items():
            if isinstance(v, dict):
                return v, k

    avail = list(criteria.keys())
    if isinstance(criteria.get("modes"), dict):
        avail += [f"modes.{k}" for k in criteria["modes"].keys()]
    raise ValueError(f"criteria 모드를 찾을 수 없습니다: '{mode}' (사용 가능: {avail})")

# ===== 임계 비교 =====
def over_threshold(scores: Dict[str, int],
                   criteria: Dict[str, Dict[str, Any]],
                   criteria_mode: str,
                   evaluation_mode: str) -> List[int]:
    th_raw, _resolved = _resolve_criteria(criteria, criteria_mode)
    th = _normalize_thresholds(th_raw, evaluation_mode)

    def is_violation(item_key: str, val: int) -> bool:
        t = th.get(item_key, None)
        if not t:
            t = {"min": None, "max": (5 if evaluation_mode == "medical" else None)}
        if t["min"] is not None and val < t["min"]:
            return True
        if t["max"] is not None and val > t["max"]:
            return True
        return False

    viol = []
    for k, v in scores.items():
        try:
            kk = str(int(k)); vv = int(v)
        except Exception:
            continue
        if is_violation(kk, vv):
            viol.append(int(kk))
    return sorted(viol)

def map_stage(violations: List[int]) -> str:
    if any(v in [1,2,3,5,7,9,12,14] for v in violations): return "content"
    if any(v in [6,10,11] for v in violations): return "both"
    if any(v in [4,8,15] for v in violations): return "content"
    return "content"

# ===== 패치 적용 =====
def apply_patches(title: str, content: str, patch_obj: Dict[str, Any]) -> Tuple[str, str]:
    new_title, new_content = title, content
    for u in patch_obj.get("patch_units", []):
        typ = u.get("type"); scope = u.get("scope")
        before = u.get("before", ""); after = u.get("after", "")
        if scope == "title":
            if typ == "replace":
                new_title = new_title.replace(before, after) if before else after
            elif typ == "insert":
                new_title = after
            elif typ == "delete" and before:
                new_title = new_title.replace(before, "")
        else:
            if typ == "replace" and before and before in new_content:
                new_content = new_content.replace(before, after)
            elif typ == "insert" and after:
                new_content += "\n\n" + after
            elif typ == "delete" and before:
                new_content = new_content.replace(before, "")
    return new_title, new_content

# ===== 프롬프트 빌드 =====
def build_eval_prompt(title: str, content: str, prompt_path: Path = EVAL_PROMPT_PATH, seo_metrics: Dict[str, int] = None) -> str:
    base = _read_text(prompt_path)
    if seo_metrics and "seo_evaluation_prompt" in str(prompt_path):
        def gs(n, v):
            if n==1:  return 12 if 26<=v<=48 else 9 if 49<=v<=69 else 6 if 15<=v<=25 else 3
            if n==2:  return 12 if 15<=v<=30 else 9 if 31<=v<=56 else 6 if 10<=v<=14 else 3
            if n==3:  return 15 if 1233<=v<=2628 else 12 if 2629<=v<=4113 else 9 if 612<=v<=1232 else 5
            if n==4:  return 15 if 936<=v<=1997 else 12 if 1998<=v<=3400 else 9 if 512<=v<=935 else 5
            if n==5:  return 10 if 249<=v<=482 else 8 if 483<=v<=672 else 6 if 183<=v<=248 else 3
            if n==6:  return 10 if 298<=v<=632 else 8 if 633<=v<=892 else 6 if 184<=v<=297 else 3
            if n==7:  return 10 if 82<=v<=193 else 8 if 194<=v<=284 else 6 if 54<=v<=81 else 3
            if n==8:  return 8 if 0<=v<=7 else 6 if 8<=v<=14 else 4 if 15<=v<=21 else 2
            if n==9:  return 8 if v >= 1 else 2
            return 0
        base += f"""

실제 측정값과 정답:
1. 제목 글자수 (공백 포함): {seo_metrics.get(1,0)}글자 → {gs(1, seo_metrics.get(1,0))}점
2. 제목 글자수 (공백 제외): {seo_metrics.get(2,0)}글자 → {gs(2, seo_metrics.get(2,0))}점
3. 본문 글자수 (공백 포함): {seo_metrics.get(3,0)}글자 → {gs(3, seo_metrics.get(3,0))}점
4. 본문 글자수 (공백 제외): {seo_metrics.get(4,0)}글자 → {gs(4, seo_metrics.get(4,0))}점
5. 총 형태소 개수: {seo_metrics.get(5,0)}개 → {gs(5, seo_metrics.get(5,0))}점
6. 총 음절 개수: {seo_metrics.get(6,0)}개 → {gs(6, seo_metrics.get(6,0))}점
7. 총 단어 개수: {seo_metrics.get(7,0)}개 → {gs(7, seo_metrics.get(7,0))}점
8. 어뷰징 단어 개수: {seo_metrics.get(8,0)}개 → {gs(8, seo_metrics.get(8,0))}점
9. 본문 이미지: {seo_metrics.get(9,0)}개 → {gs(9, seo_metrics.get(9,0))}점

위의 정답 점수를 그대로 사용하세요! 다른 점수를 부여하지 마세요!"""
    enforce = "\n\n반드시 위의 출력 형식의 JSON만 출력하고, 추가 설명은 쓰지 마십시오."
    return base.replace("[여기에 제목 입력]", title).replace("[여기에 본문 입력]", content) + enforce

def build_regen_prompt(title: str, content: str, criteria_mode: str,
                       violations: List[int], hints: List[str]) -> str:
    base = _read_text(REGEN_PROMPT_PATH)
    vnames = [f"{CHECKLIST_NAMES[i]}({i})" for i in violations]
    return (base
            .replace("{title}", title)
            .replace("{content}", content)
            .replace("{criteria}", criteria_mode)
            .replace("{violations}", json.dumps(vnames, ensure_ascii=False))
            .replace("{hints}", json.dumps(hints or [], ensure_ascii=False)))

# ===== 재생성 적합도 =====
RISK_KEYWORDS = {
    "부작용": [r"부작용", r"주의사항", r"개인차", r"합병증"],
    "가격고지": [r"가격", r"비용", r"추가\s*비용", r"부가세"],
    "근거제시": [r"연구|임상|데이터|근거|가이드라인"],
    "유인삭제": [r"리뷰\s*이벤트|추첨|사은품|기프티콘|대가"],
    "과장완화": [r"100\s*%|최고|유일|완전\s*무통|부작용\s*없"],
}

def _presence_rate(text: str, patterns: List[str]) -> float:
    return (sum(1 for p in patterns if re.search(p, text, re.I)) / len(patterns)) if patterns else 0.0

def regen_fit_score(before_over: List[int], after_over: List[int],
                    before_text: str, after_text: str,
                    tips: List[str]) -> Dict[str, Any]:
    b = len(before_over); a = len(after_over)
    risk_reduction = (b - a) / b if b else 1.0
    adherence_checks = []
    for t in tips:
        t = str(t)
        key = None
        if any(k in t for k in ["부작용","주의","개인차"]): key = "부작용"
        elif any(k in t for k in ["가격","비용","부가세"]): key = "가격고지"
        elif any(k in t for k in ["연구","임상","데이터","근거"]): key = "근거제시"
        elif any(k in t for k in ["리뷰","이벤트","추첨","사은품","대가","기프티콘"]): key = "유인삭제"
        elif any(k in t for k in ["100%","최고","유일","완전","무통","과장","절대"]): key = "과장완화"
        if key:
            pats = RISK_KEYWORDS[key]
            if key in ["부작용","가격고지","근거제시"]:
                adherence_checks.append(_presence_rate(after_text, pats))
            else:
                before_r = _presence_rate(before_text, pats); after_r = _presence_rate(after_text, pats)
                adherence_checks.append(1.0 if after_r < before_r else 0.0)
    guideline_adherence = sum(adherence_checks)/len(adherence_checks) if adherence_checks else 0.0
    def stats(s: str):
        paras = [p for p in s.split("\n\n") if p.strip()]
        sents = re.split(r"[.!?]\s+|[.\n]\s+", s)
        return {"paras": len(paras) or 1, "sents": len([x for x in sents if x.strip()]) or 1, "chars": len(s) or 1}
    sb, sa = stats(before_text), stats(after_text)
    stable = lambda a,b: max(0.0, 1.0 - abs(a-b)/max(a,1))
    flow = 0.5*stable(sa["paras"], sb["paras"]) + 0.3*stable(sa["sents"], sb["sents"]) + 0.2*stable(sa["chars"], sb["chars"])
    return {"risk_reduction_rate": round((risk_reduction),3),
            "guideline_adherence": round((guideline_adherence),3),
            "flow_stability": round((max(0.0,min(flow,1.0))),3),
            "score_0_100": round((0.5*risk_reduction + 0.3*guideline_adherence + 0.2*flow)*100)}

# ===== 에이전트 실행 래퍼 =====
def _safe_run(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print(f"[router] run failed: {' '.join(cmd)} :: {e}")
        return False

def _run_plan_agent(base_dir: Path) -> Optional[Path]:
    plan_py = ROOT / "agents" / "plan_agent.py"
    if not plan_py.exists():
        return None
    ok = _safe_run(["python", str(plan_py)])
    if not ok: return None
    cands = sorted(list(base_dir.glob("*_plan.json")), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None

def _run_title_agent(plan_path: Optional[Path]) -> Optional[str]:
    title_py = ROOT / "agents" / "title_agent.py"
    if not title_py.exists() or plan_path is None:
        return None
    ok = _safe_run(["python", str(title_py), "--plan", str(plan_path)])
    if not ok: return None
    cands = sorted(list(plan_path.parent.glob("*_title.json")), key=lambda p: p.stat().st_mtime)
    if not cands: return None
    obj = json.loads(cands[-1].read_text(encoding="utf-8"))
    return (obj.get("selected", {}) or {}).get("title", "") or None

def _run_content_agent(input_title: str, input_content: str,
                       violations: List[int], tips: List[str],
                       criteria_mode: str, evaluation_mode: str,
                       out_dir: Path) -> Tuple[str, str]:
    content_py = ROOT / "agents" / "content_agent.py"
    if content_py.exists():
        tmp_in = out_dir / f"{_nowstamp()}_content_in.json"
        tmp_in.write_text(json.dumps({
            "title": input_title,
            "content": input_content,
            "violations": violations,
            "tips": tips,
            "criteria_mode": criteria_mode,
            "evaluation_mode": evaluation_mode
        }, ensure_ascii=False), encoding="utf-8")
        _safe_run(["python", str(content_py), "--input", str(tmp_in)])
        cands = sorted(list(out_dir.glob("*_content.json")), key=lambda p: p.stat().st_mtime)
        if cands:
            obj = json.loads(cands[-1].read_text(encoding="utf-8"))
            return obj.get("title", input_title), obj.get("content", input_content)
    # Fallback: 기존 LLM 국소수정
    model = _setup_llm()
    regen_prompt = build_regen_prompt(input_title, input_content, criteria_mode, violations, tips)
    patch_obj = _call_llm(model, regen_prompt)
    return apply_patches(input_title, input_content, patch_obj)

# ===== 라우팅 판단 =====
def _needs_title_fix(seo_metrics: Dict[int,int], th_block_norm: Dict[str,Dict[str,Optional[int]]]) -> bool:
    def below_min(idx: int) -> bool:
        t = th_block_norm.get(str(idx), {})
        tmin = t.get("min", None)
        return (tmin is not None) and (seo_metrics.get(idx, 0) < tmin)
    return below_min(1) or below_min(2)

def _looks_unstructured(content: str) -> bool:
    h2s = len(re.findall(r"^##\s+", content, flags=re.M))
    paras = len([p for p in content.split("\n\n") if p.strip()])
    return (len(content.replace(" ","")) < 800) or (h2s < 2 and paras < 6)

def decide_routes(evaluation_mode: str,
                  violations: List[int],
                  th_block_norm: Dict[str,Dict[str,Optional[int]]],
                  seo_metrics: Dict[int,int],
                  title: str, content: str, analysis: str) -> List[str]:
    routes: List[str] = []
    # SEO: 제목 길이/제목 품질
    if evaluation_mode == "seo" and seo_metrics:
        if _needs_title_fix(seo_metrics, th_block_norm):
            routes.append("title")
        # 본문 관련 SEO(3~9) 미달 → content
        for idx in range(3,10):
            t = th_block_norm.get(str(idx), {})
            tmin = t.get("min", None)
            if tmin is not None and seo_metrics.get(idx, 0) < tmin:
                if "content" not in routes:
                    routes.append("content")
                break

    # 의료법: 위반 존재 시 주로 content, 제목 과장 단서 있으면 title도
    if evaluation_mode == "medical" and violations:
        if re.search(r"(100\s*%|최고|유일|완전\s*무통|부작용\s*없)", title, flags=re.I):
            routes.append("title")
        if "content" not in routes:
            routes.append("content")

    # 구성 이상 → plan 우선
    if _looks_unstructured(content) or re.search(r"(구성|섹션|체계|구조|전개)", analysis):
        routes = ["plan"] + [r for r in routes if r != "plan"]

    ordered = []
    for k in ["plan","title","content"]:
        if k in routes and k not in ordered:
            ordered.append(k)
    return ordered

# ===== 메인 루프 / Router =====
def run(criteria_mode: str = "표준",
        max_loops: int = 2,
        auto_yes: bool = False,
        log_dir: Union[str, None] = None,
        pattern: Union[str, None] = None,
        debug: bool = False,
        csv_path: Union[str, None] = None,
        report_path: Union[str, None] = None,
        evaluation_mode: str = "medical",
        mode: str = "use",
        router: bool = False,
        route_max: int = 1
        ):

    # 검색 로그 디렉토리
    log_dir_path = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 저장 디렉토리: test_logs/{mode}/{YYYYMMDD}
    mode = (mode or "use").strip().lower()
    if mode not in ("test","use"):
        mode = "use"
    save_dir = ROOT / "test_logs" / mode / _today()
    save_dir.mkdir(parents=True, exist_ok=True)

    # 탐색 패턴
    patterns = [p.strip() for p in (pattern.split(",") if pattern else []) if p.strip()]
    search_patterns = patterns or [
        "*_content.json", "*content*.json", "*_plan.json", "*_title.json",
        "*_content_log.json", "*_Content.json", "*_CONTENT.json",
        "*_title_log.json", "*title*.json", "*_plan_logs.json",
    ]

    # 0) 입력 로드
    content_path = _latest(log_dir_path, search_patterns)
    clog = _read_json(content_path)

    # plan 로그라면 실제 plan으로 스왑
    if isinstance(clog, dict) and isinstance(clog.get("output_paths"), dict):
        plan_path_str = clog["output_paths"].get("plan_path")
        if plan_path_str:
            pp = Path(plan_path_str)
            if not pp.is_absolute():
                pp = (ROOT / pp).resolve()
                if not pp.exists():
                    pp2 = (content_path.parent / Path(plan_path_str).name)
                    if pp2.exists():
                        pp = pp2
            if pp.exists():
                clog = _read_json(pp)
                content_path = pp

    # title만 있으면 plan 보강
    if isinstance(clog, dict) and "selected" in clog and "content_plan" not in clog and "plan_snapshot" not in clog:
        plan_cands = sorted(list(content_path.parent.glob("*_plan.json")), key=lambda p: p.stat().st_mtime)
        if plan_cands:
            try:
                plan_obj = _read_json(plan_cands[-1])
                clog = {"selected": clog.get("selected", {}), **plan_obj}
            except Exception:
                pass

    title, content, dbg = _extract_title_content(clog)

    if debug:
        _write_json(save_dir / f"{_nowstamp()}_debug_title_content.json", {"source": str(content_path), **dbg})

    if not title:
        top_keys = sorted(list(clog.keys())) if isinstance(clog, dict) else []
        raise ValueError(
            f"{Path(content_path).name}에서 title을 찾을 수 없습니다.\n"
            f"- 디버그: {dbg}\n- 최상위 키: {top_keys}"
        )
    if not content:
        content = "제목 평가용 더미 콘텐츠입니다."

    # SEO 메트릭
    seo_metrics: Dict[int,int] = {}
    if evaluation_mode == "seo":
        seo_metrics = calculate_seo_metrics(title, content)

    # 1) 기준/CSV/리포트
    if evaluation_mode == "seo":
        criteria = _read_json(SEO_CRITERIA_PATH); eval_prompt_path = SEO_PROMPT_PATH
        rule_all: Dict[str, Dict[str, Any]] = {}; weights = {str(i): 1.0 for i in range(1, 10)}
    else:
        criteria = _read_json(CRITERIA_PATH); eval_prompt_path = EVAL_PROMPT_PATH
        csv_file = Path(csv_path) if csv_path else _find_existing(DEFAULT_CSV_PATHS)
        rows = load_checklist_csv(csv_file)
        pats = compile_patterns(rows)
        report_file = Path(report_path) if report_path else _find_existing(DEFAULT_REPORT_PATHS)
        weights = parse_report_weights(report_file)
        rule_all = rule_score_all(title, content, pats)

    # 2) 1차 평가
    model = _setup_llm()
    eval_prompt = build_eval_prompt(title, content, eval_prompt_path, seo_metrics if evaluation_mode=="seo" else None)
    result = _call_llm(model, eval_prompt)
    llm_scores: Dict[str, int] = result.get("평가결과", {}) or {}
    analysis: str = result.get("상세분석", "") or ""
    tips: List[str] = result.get("권고수정", []) or []

    # 3) 스코어 융합
    def fuse(rule_all_: Dict[str, Dict[str,Any]], llm_scores_: Dict[str,int], upper:int) -> Dict[str,int]:
        fused = {}
        for i in range(1, upper):
            r = int(rule_all_.get(str(i),{}).get("score",0))
            l = int(llm_scores_.get(str(i),0))
            fused[str(i)] = max(r,l)
        return fused
    upper = 10 if evaluation_mode == "seo" else 16
    final_scores = fuse(rule_all, llm_scores, upper)

    # 임계/위반
    violations = over_threshold(final_scores, criteria, criteria_mode, evaluation_mode)
    th_raw, th_name = _resolve_criteria(criteria, criteria_mode)
    th_norm = _normalize_thresholds(th_raw, evaluation_mode)

    # ===== Router 모드 =====
    if router:
        base_dir = content_path.parent
        routes = decide_routes(evaluation_mode, violations, th_norm, seo_metrics, title, content, analysis)
        routed: List[str] = []

        for stage in routes:
            for _ in range(max(1, route_max)):
                if stage == "plan":
                    plan_latest = _run_plan_agent(base_dir)
                    if plan_latest:
                        plan_obj = _read_json(plan_latest)
                        new_content = _plan_to_text(plan_obj)
                        if new_content.strip():
                            content = new_content
                    routed.append("plan")
                    break

                elif stage == "title":
                    plan_cands = sorted(list(base_dir.glob("*_plan.json")), key=lambda p: p.stat().st_mtime)
                    plan_path = plan_cands[-1] if plan_cands else None
                    fixed = _run_title_agent(plan_path)
                    if fixed:
                        title = fixed
                    routed.append("title")
                    break

                elif stage == "content":
                    title, content = _run_content_agent(title, content, violations, tips,
                                                        criteria_mode, evaluation_mode, base_dir)
                    routed.append("content")
                    break

        # 라우팅 후 재평가
        if evaluation_mode == "seo":
            seo_metrics = calculate_seo_metrics(title, content)
            eval_prompt = build_eval_prompt(title, content, SEO_PROMPT_PATH, seo_metrics)
            rule_all = {}
            weights = {str(i): 1.0 for i in range(1,10)}
        else:
            eval_prompt = build_eval_prompt(title, content, EVAL_PROMPT_PATH)
            rule_all = rule_score_all(title, content, pats)

        model = _setup_llm()
        result = _call_llm(model, eval_prompt)
        llm_scores = result.get("평가결과", {}) or {}
        analysis = result.get("상세분석", "") or ""
        tips = result.get("권고수정", []) or []

        upper = 10 if evaluation_mode == "seo" else 16
        final_scores = fuse(rule_all, llm_scores, upper)

        violations = over_threshold(final_scores, criteria, criteria_mode, evaluation_mode)
        th_raw, th_name = _resolve_criteria(criteria, criteria_mode)
        th_norm = _normalize_thresholds(th_raw, evaluation_mode)

        # 결과 저장(루프 없음)
        by_item = {}
        for i in range(1, upper):
            fi = str(i)
            tmin = th_norm.get(fi, {}).get("min", None)
            tmax = th_norm.get(fi, {}).get("max", (5 if evaluation_mode=="medical" else None))
            val = int(final_scores.get(fi, 0))
            passed = True
            if tmin is not None and val < tmin: passed = False
            if tmax is not None and val > tmax: passed = False
            by_item[fi] = {
                "name": (SEO_CHECKLIST_NAMES[i] if evaluation_mode=="seo" else CHECKLIST_NAMES[i]),
                "rule_score": int(rule_all.get(fi, {}).get("score", 0)),
                "llm_score": int(llm_scores.get(fi, 0)),
                "final_score": val,
                "threshold_min": tmin,
                "threshold_max": tmax,
                "passed": passed,
                "evidence": {"regex_hits": rule_all.get(fi, {}).get("hits", [])},
                **({"actual_value": seo_metrics.get(i, 0)} if evaluation_mode=="seo" else {})
            }

        out = {
            "input": {"source_log": Path(content_path).name, "title": title, "content": content, "content_len": len(content)},
            "modes": {"criteria": criteria_mode, "evaluation": evaluation_mode, "mode": mode, "router": True},
            "scores": {
                "by_item": by_item,
                "weighted_total": weighted_total(final_scores, weights, evaluation_mode),
                "llm_total_raw": sum(int(llm_scores.get(str(i),0)) for i in range(1, upper)),
                "rule_total_proxy": sum(int(rule_all.get(str(i),{}).get("score",0)) for i in range(1, upper))
            },
            "violations": {"over_threshold": violations,
                           "names": [(SEO_CHECKLIST_NAMES[i] if evaluation_mode=="seo" else CHECKLIST_NAMES[i]) for i in violations]},
            "routed": routed,
            "title": title,
            "content": content
        }
        out_path = save_dir / f"{_nowstamp()}_evaluation.json"
        _write_json(out_path, out)
        print(("✅ 기준 충족. " if not violations else "⚠️ 일부 미충족. ") + f"Router 결과 저장: {out_path.name}")
        return

    # ===== 비Router 모드 (기존 while 루프: fallback) =====
    history: List[Dict[str, Any]] = []
    loop = 0
    patched_once = False
    title_before, content_before = title, content

    while True:
        loop += 1
        history.append({
            "loop": loop,
            "rule_scores": {k:v["score"] for k,v in rule_all.items()},
            "llm_scores": llm_scores,
            "final_scores": final_scores,
            "violations": violations,
            "analysis": analysis,
            "tips": tips
        })

        weighted_total_now = weighted_total(final_scores, weights, evaluation_mode)

        if not violations or loop >= max_loops:
            # by_item 구성
            by_item = {}
            for i in range(1, upper):
                fi = str(i)
                tmin = th_norm.get(fi, {}).get("min", None)
                tmax = th_norm.get(fi, {}).get("max", (5 if evaluation_mode=="medical" else None))
                val = int(final_scores.get(fi, 0))
                passed = True
                if tmin is not None and val < tmin: passed = False
                if tmax is not None and val > tmax: passed = False
                by_item[fi] = {
                    "name": (SEO_CHECKLIST_NAMES[i] if evaluation_mode == "seo" else CHECKLIST_NAMES[i]),
                    "rule_score": int(rule_all.get(fi, {}).get("score", 0)),
                    "llm_score": int(llm_scores.get(fi, 0)),
                    "final_score": val,
                    "threshold_min": tmin,
                    "threshold_max": tmax,
                    "passed": passed,
                    "evidence": {"regex_hits": rule_all.get(fi, {}).get("hits", [])},
                    **({"actual_value": seo_metrics.get(i, 0)} if evaluation_mode == "seo" else {})
                }

            out = {
                "input": {"source_log": Path(content_path).name, "title": title, "content": content, "content_len": len(content)},
                "modes": {"criteria": criteria_mode, "evaluation": evaluation_mode, "mode": mode, "router": False},
                "scores": {
                    "by_item": by_item,
                    "weighted_total": weighted_total_now,
                    "llm_total_raw": sum(int(llm_scores.get(str(i),0)) for i in range(1, upper)),
                    "rule_total_proxy": sum(int(rule_all.get(str(i),{}).get("score",0)) for i in range(1, upper))
                },
                "violations": {"over_threshold": violations,
                               "names": [(SEO_CHECKLIST_NAMES[i] if evaluation_mode == "seo" else CHECKLIST_NAMES[i]) for i in violations]},
                "regen_fit": {"applied": patched_once},
                "notes": {"recommendations": tips},
                "title": title,
                "content": content
            }

            if patched_once:
                b_over = history[0]["violations"]; a_over = violations
                before_text = f"{title_before}\n\n{content_before}"
                after_text  = f"{title}\n\n{content}"
                rf = regen_fit_score(b_over, a_over, before_text, after_text, tips)
                out["regen_fit"].update({
                    "before_over_threshold": len(b_over),
                    "after_over_threshold": len(a_over),
                    **rf
                })

            out_path = save_dir / f"{_nowstamp()}_evaluation.json"
            _write_json(out_path, out)
            print(("✅ 기준 충족. " if not violations else "⚠️ 반복 상한 도달. ") + f"결과 저장: {out_path.name}")
            return

        # 필요 시 재생성 (비Router fallback)
        if not auto_yes:
            yn = input(f"기준 초과 항목 {violations}가 있습니다. 국소 수정 진행할까요? (Y/n): ").strip().lower()
            if yn and yn.startswith("n"):
                break

        # 재생성 → 패치
        regen_prompt = build_regen_prompt(title, content, criteria_mode, violations, tips)
        patch_obj = _call_llm(model, regen_prompt)
        title, content = apply_patches(title, content, patch_obj)
        patched_once = True

        # 재평가
        if evaluation_mode == "medical":
            rule_all = rule_score_all(title, content, pats)
        else:
            rule_all = {}
            seo_metrics = calculate_seo_metrics(title, content)

        eval_prompt = build_eval_prompt(title, content, eval_prompt_path, seo_metrics if evaluation_mode=="seo" else None)
        result = _call_llm(model, eval_prompt)
        llm_scores = result.get("평가결과", {}) or {}
        analysis = result.get("상세분석", "") or ""
        tips = result.get("권고수정", []) or []

        final_scores = fuse(rule_all, llm_scores, upper)
        violations = over_threshold(final_scores, criteria, criteria_mode, evaluation_mode)
        th_raw, th_name = _resolve_criteria(criteria, criteria_mode)
        th_norm = _normalize_thresholds(th_raw, evaluation_mode)

# ===== CLI =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", default="표준", help="엄격 | 표준 | 유연 (또는 우수 | 양호 | 보통, 또는 strict | standard | lenient)")
    parser.add_argument("--max_loops", type=int, default=2)
    parser.add_argument("--auto-yes", action="store_true")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="로그 디렉토리(검색용, 기본: test_logs/test)")
    parser.add_argument("--pattern", default="", help="탐색 패턴(쉼표로 여러 개). 비우면 기본 패턴 리스트 사용")
    parser.add_argument("--debug", action="store_true", help="추출 후보/경로 디버그 로그 저장")
    parser.add_argument("--csv-path", default="", help="medical_ad_checklist.csv 경로(미지정 시 기본 경로/ /mnt/data 탐색)")
    parser.add_argument("--report-path", default="", help="medical-ad-report.md 경로(미지정 시 기본 경로/ /mnt/data 탐색)")
    parser.add_argument("--evaluation-mode", default="medical", choices=["medical", "seo"], help="평가 모드 (medical: 의료법, seo: SEO 품질)")
    parser.add_argument("--mode", default="use", choices=["test","use"], help="저장 모드 (test/use) — 결과 저장 경로를 test_logs/{mode}/ 로 고정")
    parser.add_argument("--router", action="store_true", help="평가 실패 시 각 스테이지 에이전트(plan/title/content)로 라우팅")
    parser.add_argument("--route-max", type=int, default=1, help="각 에이전트를 최대 몇 번까지 재시도할지 (기본 1회)")
    args = parser.parse_args()

    run(criteria_mode=args.criteria,
        max_loops=args.max_loops,
        auto_yes=args.auto_yes,
        log_dir=args.log_dir,
        pattern=args.pattern,
        debug=args.debug,
        csv_path=(args.csv_path or None),
        report_path=(args.report_path or None),
        evaluation_mode=args.evaluation_mode,
        mode=args.mode,
        router=args.router,
        route_max=args.route_max)
