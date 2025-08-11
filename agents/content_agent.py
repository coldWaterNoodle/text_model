# agents/content_agent.py
# -*- coding: utf-8 -*-
"""
ContentAgent (7개 섹션 프롬프트 + 취합 스티처)
- 입력: 최신 *_input_log.json / *_plan_log.json / *_title_log.json
- 섹션: content1~7_*_prompt.txt 각각 호출 → 섹션별 후보 생성(JSON) → 섹션별 최종 선택(JSON)
- 스티치: content_stitch_prompt.txt 로 7개 섹션을 한 편으로 통합(중복 제거·흐름 정리)
- 후처리: 타 병원/의사명 제거, 금지어 필터, homepage 링크 1회, 안내 박스 1회 유지
- 출력/저장:
  - *_content_sections_log.json (섹션별 후보/선정 원본)
  - *_content_log.json          (최종 합본/컨텍스트 메타)
  - *_content.md                (최종 마크다운)
  - *_content_full.txt          (TITLE + 최종 마크다운)
"""

import os, sys, re, json, textwrap
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import google.generativeai as genai
from dotenv import load_dotenv

# ---------------- 기본 세팅 ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 필요합니다(.env)")
genai.configure(api_key=API_KEY)

sys.path.append(str(Path(".").resolve()))

# ---------------- 유틸 ----------------
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

def latest_input_log(dir_: Path) -> Optional[Path]:
    return latest_file_by_mtime(dir_, "*_input_log.json")

def latest_plan_log(dir_: Path) -> Optional[Path]:
    # 실제 plan 내용은 *_plan_log.json
    return latest_file_by_mtime(dir_, "*_plan_log.json")

def latest_title_log(dir_: Path) -> Optional[Path]:
    return latest_file_by_mtime(dir_, "*_title_log.json")

def _pick(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# ---------------- JSON 추출 ----------------
JSON_BLOCK_RE = re.compile(r"\s*```(?:json)?\s*(\{.*?\})\s*```|\s*(\{.*\})\s*", re.S)
def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("빈 모델 응답")
    m = JSON_BLOCK_RE.search(text)
    blob = (m.group(1) or m.group(2)) if m else None
    if not blob:
        s = text.find("{")
        if s == -1:
            raise ValueError("JSON 시작 '{' 없음")
        stack = 0; e = -1
        for i, ch in enumerate(text[s:], s):
            if ch == "{": stack += 1
            elif ch == "}":
                stack -= 1
                if stack == 0:
                    e = i + 1
                    break
        if e == -1:
            raise ValueError("JSON 중괄호 불일치")
        blob = text[s:e]
    return json.loads(blob)

# ---------------- Gemini ----------------
class GeminiClient:
    def __init__(self, model="gemini-1.5-pro", temperature=0.65, max_output_tokens=8192):
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

# ---------------- 섹션 프롬프트 7개 ----------------
SECTION_FILES = [
    ("intro",        "content1_intro_prompt.txt"),
    ("visit",        "content2_visit_prompt.txt"),
    ("inspection",   "content3_inspection_prompt.txt"),
    ("doctor_tip",   "content4_doctor_tip_prompt.txt"),
    ("treatment",    "content5_treatment_prompt.txt"),
    ("check_point",  "content6_check_point_prompt.txt"),
    ("conclusion",   "content7_conclusion_prompt.txt"),
]

class PromptPack:
    def __init__(self):
        base = Path("test_prompt")
        if not base.exists():
            base = Path("prompts")
        self.dir = base
        self.paths: Dict[str, Path] = {key: base / fname for key, fname in SECTION_FILES}
        self.stitch_path = self.dir / "content_stitch_prompt.txt"

    def _fallback(self, key: str) -> str:
        return textwrap.dedent(f"""
        아래 '컨텍스트'를 반영해 블로그 본문의 '{key}' 섹션을 생성하세요.
        - 의료광고법 위반 표현 금지(과장/단정/가격/비교사례/치료경험담)
        - 줄바꿈/문장 길이/느낌표 사용은 자연스럽게
        출력 형식(JSON 하나):
        {{
          "candidates": [{{"id":"cand_1","style":"...","content_markdown":"..."}}],
          "selected": {{"id":"cand_1","why_best":"...","content_markdown":"..."}}
        }}

        [컨텍스트]
        title={{title}}
        plan={{plan}}
        input_data={{input_data}}
        vars={{vars}}
        N={{N}}
        """).strip()

    def load(self, key: str) -> str:
        p = self.paths[key]
        return read_text(p) if p.exists() else self._fallback(key)

    def fill(self, template: str, *, N: int, title: str, plan: Dict[str, Any],
             input_data: Dict[str, Any], vars_: Dict[str, Any]) -> str:
        return (template
            .replace("{title}", json.dumps(title, ensure_ascii=False))
            .replace("{plan}", json.dumps(plan, ensure_ascii=False))
            .replace("{input_data}", json.dumps(input_data, ensure_ascii=False))
            .replace("{vars}", json.dumps(vars_, ensure_ascii=False))
            .replace("{N}", str(N)))

    def stitch_template(self) -> str:
        if self.stitch_path.exists():
            return read_text(self.stitch_path)
        # 안전 fallback (파일 없을 때)
        return (
            "당신은 대한민국 치과 블로그 전문 편집자입니다. 아래 자료를 바탕으로 "
            "네이버 블로그용 '최종 본문(markdown 하나)'을 완성하세요.\n\n"
            "[목표]\n"
            "- 중복/반복/군더더기 제거, 흐름 매끄럽게 정리\n"
            "- 페르소나 톤 유지\n"
            "- 의료광고법 위반 표현 금지(과장/단정/가격/치료경험담/비교사진)\n"
            "- 병원/링크 정책 준수: homepage 본문 1회만, map은 말미 안내 박스\n"
            "- 이미지 참조 문구(사진: *)는 필요한 곳에 1회씩만\n\n"
            "TITLE: {title}\n\n"
            "CONTEXT(JSON):\n{context_json}\n\n"
            "SECTIONS(JSON):\n{sections_json}\n\n"
            "PLAN(JSON):\n{plan_json}\n\n"
            "[출력 규칙]\n"
            "- 마크다운 본문 하나만 출력(# 제목 포함)\n"
            "- 인트로 1개, 결론 1개, 안내 박스는 문서 맨 끝 1회만\n"
            "- 외부 병원/의사명 언급 금지, clinic_name 표기 일관\n"
            "- 홈페이지 링크는 본문 중간 1회만(이미 있으면 추가 금지)\n"
            "지금부터 최종본만 출력하세요."
        )

# ---------------- 페르소나 → 가이드 ----------------
def _persona_style(selected: List[str]) -> Dict[str, Any]:
    s = " ".join(selected or []).lower()
    guide = {
        "tone": "친근하고 담백한 정보 전달",
        "reading_aids": ["짧은 문장", "줄바꿈 자주", "목록으로 핵심 정리"],
        "focus": "검진 필요성, 관리 요령, 과장 없는 설명",
        "include_blocks": []
    }
    if "직장" in s or "바빠" in s or "워킹" in s:
        guide.update({
            "tone": "바쁜 직장인에게 간결하게",
            "reading_aids": ["요약 먼저", "체크리스트", "내원 시간/횟수 명시"],
            "focus": "진료 동선·시간·통증 관리",
            "include_blocks": ["요약박스"]
        })
    if "보호자" in s or "자녀" in s or "학부모" in s:
        guide.update({
            "tone": "부모에게 안심 주는 설명",
            "reading_aids": ["단계별 안내", "주의사항 강조"],
            "focus": "아이 협조·통증·사후관리"
        })
    if "시니어" in s or "노년" in s:
        guide.update({
            "tone": "또박또박 쉬운 용어",
            "reading_aids": ["큰 단락 구분", "용어 풀이"],
            "focus": "복약/전신질환/의치 고려"
        })
    if "대학" in s or "학생" in s:
        guide.update({
            "tone": "친근하고 캐주얼",
            "reading_aids": ["Q&A", "이모지 제한적 사용"],
            "focus": "비용 언급 없이 관리 루틴/습관"
        })
    if "임산부" in s:
        guide.update({
            "tone": "안전·시기별 유의점 중심",
            "reading_aids": ["금기/권고 구분"],
            "focus": "방사선 노출 회피/응급 시 대처"
        })
    return guide

# ---------------- Stitcher ----------------
class Stitcher:
    def __init__(self, prompts: PromptPack, model="gemini-1.5-pro"):
        self.prompts = prompts
        self.gemini = GeminiClient(model=model)

    def stitch(self, *, title: str, context: Dict[str, Any],
               sections_out: Dict[str, Dict[str, Any]], plan: Dict[str, Any]) -> str:
        sections_json = json.dumps({
            k: (v.get("selected") or {}).get("content_markdown", "")
            for k, v in sections_out.items()
        }, ensure_ascii=False, indent=2)
        tpl = self.prompts.stitch_template()
        prompt = (
            tpl.replace("{title}", title)
               .replace("{context_json}", json.dumps(context, ensure_ascii=False, indent=2))
               .replace("{sections_json}", sections_json)
               .replace("{plan_json}", json.dumps(plan, ensure_ascii=False, indent=2))
        )
        raw = self.gemini.generate_text(prompt, temperature=0.55)
        return (raw or "").strip()

# ---------------- ContentAgent ----------------
class ContentAgent:
    def __init__(self, model="gemini-1.5-pro"):
        self.prompts = PromptPack()
        self.gemini = GeminiClient(model=model)

    # 질문 8개 유연 추출
    def _extract_questions(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        q = {}
        if isinstance(input_data.get("questions"), list):
            for i, item in enumerate(input_data["questions"], 1):
                q[f"q{i}"] = item
        for k, v in list(input_data.items()):
            if isinstance(k, str) and k.lower().startswith("question"):
                q[k] = v
        for k in [
            "question1_concept","question2_condition","question3_visit",
            "question4_treatment","question5_therapy","question6_result",
            "question7_followup","question8_extra"
        ]:
            if k in input_data:
                q[k] = input_data[k]
        return q

    # 컨텍스트 병합
    def _build_context(self, plan: Dict[str, Any], title: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        hospital  = dict(input_data.get("hospital", {}))
        meta      = plan.get("meta_panel", {}) or {}
        geo       = plan.get("geo_branding", {}) or {}
        linkpol   = plan.get("link_policy", {}) or {}
        images_ix = input_data.get("images_index", {}) or {}
        personas  = input_data.get("selected_personas", []) or input_data.get("persona_candidates", [])
        persona_guide = _persona_style(personas)
        ctx = {
            "clinic_name": _pick(hospital.get("name"), geo.get("clinic_alias"), "하니치과"),
            "doctor_name": "김하니",  # 내부 변수. 본문 노출 금지
            "region_phrase": _pick(geo.get("region_line"), hospital.get("region_phrase")),
            "address": _pick(meta.get("address"), hospital.get("address")),
            "phone": _pick(meta.get("phone"), hospital.get("phone")),
            "homepage": _pick(meta.get("homepage"), hospital.get("homepage")),
            "map_link": _pick(meta.get("map_link"), hospital.get("map_link")),
            "link_policy": linkpol,
            "category": input_data.get("category",""),
            "symptom": input_data.get("symptom","") or input_data.get("question2_condition",""),
            "diagnosis": input_data.get("diagnosis",""),
            "treatment": input_data.get("treatment","") or input_data.get("question4_treatment",""),
            "procedure": input_data.get("procedure",""),
            "questions": self._extract_questions(input_data),
            "personas": personas,
            "persona_guide": persona_guide,
            "title": title,
            "question3_visit_photo":   images_ix.get("question3_visit_photo",   "visit_images:0"),
            "question5_therapy_photo": images_ix.get("question5_therapy_photo", "therapy_images:0"),
            "question6_result_photo":  images_ix.get("question6_result_photo",  "result_images:0"),
        }
        return ctx

    def _context_header(self, context: Dict[str, Any]) -> str:
        safe = json.dumps(context, ensure_ascii=False, indent=2)
        rules = (
            "- CONTEXT의 clinic_name/address/homepage/map_link/link_policy/카테고리/증상·진료·치료/질문/페르소나 반영.\n"
            "- doctor_name은 본문 노출 금지(내부 변수용).\n"
            "- persona_guide의 tone/reading_aids를 문장 스타일과 구성에 반영.\n"
            "- 의료광고법: 과장/단정/가격/비교사례/치료경험담 금지.\n"
            "- homepage 링크는 link_policy.homepage_in_body_once가 true일 때 본문 1회만, map_link는 말미 안내 박스에만.\n"
            "- 예시 속 타 병원/의사명(동탄 내이튼 치과/윤민정 등) 금지. clinic_name 표기 일관 유지.\n"
        )
        return f"### CONTEXT (DO NOT IGNORE)\n{safe}\n\n### RULES\n{rules}\n"

    def _gen_section(self, key: str, *, title: str, plan: Dict[str, Any],
                     input_data: Dict[str, Any], N: int) -> Dict[str, Any]:
        tmpl = self.prompts.load(key)
        context = self._build_context(plan, title, input_data)
        # 예시 플레이스홀더 치환
        for ph in ["question3_visit_photo", "question5_therapy_photo", "question6_result_photo"]:
            tmpl = tmpl.replace(f"{{{ph}}}", context.get(ph, ""))
        # 출력 스키마 강제
        enforce = """
        출력은 반드시 다음 JSON 하나로만 제공하세요:
        {
          "candidates":[{"id":"cand_1","style":"...","content_markdown":"..."}],
          "selected":{"id":"cand_1","why_best":"...","content_markdown":"..."}
        }
        """.strip()
        prompt_core = self.prompts.fill(tmpl + "\n\n" + enforce,
                                        N=N, title=title, plan=plan, input_data=input_data, vars_=context)
        prompt = self._context_header(context) + "\n" + prompt_core
        raw = self.gemini.generate_text(prompt, temperature=0.65)
        obj = extract_json(raw)
        obj.setdefault("candidates", []); obj.setdefault("selected", {})
        if not obj["selected"].get("content_markdown") and obj["candidates"]:
            obj["selected"] = {
                "id": obj["candidates"][0].get("id", "cand_1"),
                "why_best": "모델 미선정 → 1순위 후보 폴백",
                "content_markdown": obj["candidates"][0].get("content_markdown", "")
            }
        return obj

    # 로컬 후처리(중복/링크/안내/금지어)
    def _postprocess_markdown(self, md: str, context: Dict[str, Any]) -> str:
        if not md: return md
        clinic = context.get("clinic_name","하니치과")

        # 제목(# ) 다중 발생 시 첫 줄만 유지, 나머지는 ## 로 다운그레이드
        lines = md.splitlines()
        if lines and lines[0].lstrip().startswith("# "):
            for i in range(1, len(lines)):
                if lines[i].lstrip().startswith("# "):
                    lines[i] = "#" + lines[i]
        md = "\n".join(lines)

        # 타 병원/의사 제거/치환
        md = re.sub(r"동탄\s*내이튼\s*치과", clinic, md)
        md = re.sub(r"윤\s*민\s*정\s*원장", "", md)

        # 과잉 공백 정리
        md = re.sub(r"(?:\n\s*){3,}", "\n\n", md)

        # 이미지 참조 중복(같은 줄 반복) 제거
        md = re.sub(r"(사진:\s*(visit|therapy|result)_images:\d+)\s*(\n\1)+", r"\1", md)

        # 홈페이지 링크 1회 정책
        homepage = context.get("homepage","")
        if homepage:
            occ = len(re.findall(re.escape(homepage), md))
            if occ == 0:
                ls = md.splitlines()
                insert_at = min(12, len(ls))
                ls.insert(insert_at, f"[자세한 안내는 홈페이지에서 확인하세요]({homepage})")
                md = "\n".join(ls)
            elif occ > 1:
                # 두 번째 이후 제거
                pattern = re.compile(rf"\[.*?\]\({re.escape(homepage)}\)")
                count = 0
                def _keep_first(m):
                    nonlocal count
                    count += 1
                    return m.group(0) if count == 1 else ""
                md = pattern.sub(_keep_first, md)

        # 안내 박스가 여러 번이면 모두 제거 후 맨 끝에 1회 재생성
        md = re.sub(r"(?s)(?:^|\n)---\n\*\*안내\*\*.*?(?=\n---|\Z)", "", md, flags=re.MULTILINE)
        info = {
            "address": context.get("address",""),
            "phone": context.get("phone",""),
            "homepage": context.get("homepage",""),
            "map_link": context.get("map_link",""),
        }
        if any(info.values()):
            tail = ["", "---", "**안내**"]
            if info["address"]: tail.append(f"- 주소: {info['address']}")
            if info["phone"]:   tail.append(f"- 전화: {info['phone']}")
            if info["homepage"]:tail.append(f"- 홈페이지: {info['homepage']}")
            if info["map_link"]:tail.append(f"- 지도: {info['map_link']}")
            tail.append("")
            md = md.rstrip() + "\n" + "\n".join(tail)

        # 금지어/과장 표현 간단 필터
        for pat in [r"100%\s*완치", r"유일한\s*치료", r"부작용\s*없"]:
            md = re.sub(pat, "", md, flags=re.I)

        return md.strip() + "\n"

    # 전체 생성
    def generate(self, *, plan: Dict[str, Any], title: str,
                 input_data: Dict[str, Any], n_candidates_each: int = 3) -> Dict[str, Any]:
        section_keys = [k for k, _ in SECTION_FILES]

        # 1) 섹션별 생성
        sections_out: Dict[str, Dict[str, Any]] = {}
        for key in section_keys:
            sections_out[key] = self._gen_section(
                key, title=title, plan=plan, input_data=input_data, N=n_candidates_each
            )

        # 2) 스티치 프롬프트로 한 편으로 통합
        context = self._build_context(plan, title, input_data)
        stitcher = Stitcher(self.prompts)
        stitched_md = stitcher.stitch(title=title, context=context, sections_out=sections_out, plan=plan)

        # 3) 로컬 후처리
        final_md = self._postprocess_markdown(stitched_md, context)

        return {
            "sections": sections_out,
            "stitched": {
                "id": "stitched_v1",
                "why_best": "Stitcher 통합 + 로컬 후처리",
                "content_markdown": final_md
            },
            "context_used": context,
            "title": title
        }

# ---------------- 저장 & 실행 ----------------
def save_all(base_dir: Path, plan_path: str, title_path: str,
             mode_label: str, result: Dict[str, Any]) -> None:
    ts = now_str()
    # 섹션 로그
    write_json(base_dir / f"{ts}_content_sections_log.json", result.get("sections", {}))
    # 합본 로그
    meta = {
        "mode": mode_label,
        "plan_path": plan_path,
        "title_path": title_path,
        "ts": ts,
        "selected_id": result.get("stitched", {}).get("id", "stitched_v1"),
        "selected_len_words": len((result.get("stitched", {}).get("content_markdown","")).split()),
    }
    write_json(base_dir / f"{ts}_content_log.json", {
        "meta": meta,
        "stitched": result.get("stitched", {}),
        "context_used": result.get("context_used", {}),
        "title": result.get("title","")
    })
    # md 저장
    md = (result.get("stitched") or {}).get("content_markdown","")
    (base_dir / f"{ts}_content.md").write_text(md, encoding="utf-8")
    # title + full content 저장
    full_txt = (result.get("title","") or "").strip() + "\n\n" + md
    (base_dir / f"{ts}_content_full.txt").write_text(full_txt, encoding="utf-8")
    print(f"✅ 저장: {base_dir/(ts+'_content_sections_log.json')}")
    print(f"✅ 저장: {base_dir/(ts+'_content_log.json')}")
    print(f"✅ 저장: {base_dir/(ts+'_content.md')}")
    print(f"✅ 저장: {base_dir/(ts+'_content_full.txt')}")

def _load_required_logs(base_dir: Path) -> Tuple[Dict[str, Any], str, Dict[str, Any], Path, Path]:
    plan_path  = latest_plan_log(base_dir)
    title_path = latest_title_log(base_dir)
    input_path = latest_input_log(base_dir)
    if not plan_path:
        raise FileNotFoundError(f"plan 로그가 없습니다: {base_dir}/*_plan_log.json")
    if not title_path:
        raise FileNotFoundError(f"title 로그가 없습니다: {base_dir}/*_title_log.json")
    if not input_path:
        raise FileNotFoundError(f"input 로그가 없습니다: {base_dir}/*_input_log.json")

    plan  = read_json(plan_path)
    title_obj = read_json(title_path)
    input_data = read_json(input_path)

    # title 추출: selected.title 우선
    title = (title_obj.get("selected") or {}).get("title") \
            or (title_obj.get("candidates", [{}])[0] or {}).get("title", "")
    if not title:
        raise ValueError("title 로그에서 제목을 찾지 못했습니다.")

    return plan, title, input_data, plan_path, title_path

def run_mode(base_dir: Path, mode_label: str) -> Optional[Dict[str, Any]]:
    try:
        plan, title, input_data, plan_path, title_path = _load_required_logs(base_dir)
    except Exception as e:
        print("❌ 준비 오류:", e)
        return None

    agent = ContentAgent()
    result = agent.generate(plan=plan, title=title, input_data=input_data, n_candidates_each=3)
    save_all(base_dir, str(plan_path), str(title_path), mode_label, result)

    # 콘솔에도 TITLE + FULL CONTENT 출력
    print("\n" + "="*80)
    print("TITLE + FULL CONTENT")
    print("="*80 + "\n")
    print(result.get("title","").strip())
    print()
    print((result.get("stitched") or {}).get("content_markdown",""))
    print("="*80 + "\n")
    return result

def run_test() -> Optional[Dict[str, Any]]:
    base_dir = Path("test_logs/test")
    return run_mode(base_dir, "test")

def run_use() -> Optional[Dict[str, Any]]:
    base_dir = Path("test_logs/use")
    return run_mode(base_dir, "use")

def run_latest_content_only(base_dir: Path) -> Optional[Dict[str, Any]]:
    latest = latest_file_by_mtime(base_dir, "*_content_log.json")
    if not latest:
        print(f"⚠️ 최신 content 결과가 없습니다: {base_dir}/*_content_log.json")
        meta = latest_file_by_mtime(base_dir, "*_content.json")
        if meta:
            print(f"ℹ️ 참고: 최신 메타 파일은 있습니다 → {meta.name}")
        return None
    obj = read_json(latest)
    sel_md = (obj.get("stitched") or {}).get("content_markdown", "")
    print(f"📄 최신 CONTENT: {latest.name}")
    print("길이(단어수):", len(sel_md.split()))
    return obj

def main():
    print("$ python agents/content_agent.py")
    print("🧾 ContentAgent (7섹션+스티처) 시작\n")
    print("모드:")
    print("1) test        → 최신 plan/title/input 로그 기반 생성 + title+풀content 저장/출력")
    print("2) use         → 최신 plan/title/input 로그 기반 생성 + title+풀content 저장/출력")
    print("3) latest-view → 최신 content 로그 요약 보기(폴더 선택)")
    sel = input("선택 (1/2/3): ").strip()
    try:
        if sel == "1":
            run_test()
        elif sel == "2":
            run_use()
        elif sel == "3":
            which = input("폴더 선택 (1: test_logs/test, 2: test_logs/use): ").strip()
            base = Path("test_logs/test") if which == "1" else Path("test_logs/use")
            run_latest_content_only(base)
        else:
            print("⚠️ 잘못된 입력"); sys.exit(1)
    except Exception as e:
        print("❌ 오류:", e); sys.exit(1)

if __name__ == "__main__":
    main()
