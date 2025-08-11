# agents/plan_agent.py
# -*- coding: utf-8 -*-
"""
PlanAgent (섹션5 고정·모델 자가보완 리파인판, 문자열 JSON 방어 포함)
- 후보 plan 생성 후, '리파인 프롬프트'로 정확히 5개 섹션(서론/진단/치료/결과/관리) 구성
- 일부 입력이 비어도 모델이 문맥으로 보완 (강제 더미 텍스트 없음)
- 1차 리파인 실패 시 2차 엄격 리파인 재시도
- test: test_logs/test의 최신 *_input_log.json 기반(없으면 1회 생성 여부 질문)
- use: ① 기존 선택(로그) ② 직접 입력(run_use) → n-way 생성
- 저장: *_plan.json / *_plan_log.json
- 최신 파일 탐색: 수정시간(mtime) 기준
"""

import os, re, sys, json, random
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# deps
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

try:
    sys.path.append(str(Path(".").resolve()))
    from input_agent import InputAgent  # noqa
except Exception:
    InputAgent = None


# ---------------- utils ----------------
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

# 안전 변환기: 모델이 문자열(JSON)로 줄 때 dict로 강제
def _coerce_dict(obj) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}


# JSON extract
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


# ---------------- Region Extractor (KR) ----------------
CITY_SHORT = {
    "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
    "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산", "세종특별자치시": "세종",
    "제주특별자치도": "제주", "강원특별자치도": "강원",
}
PROVINCES = ["경기도","충청북도","충청남도","전라북도","전라남도","경상북도","경상남도","강원특별자치도","제주특별자치도"]
METROS = list(CITY_SHORT.keys())

def extract_region_from_address(addr: str) -> Tuple[str, str, str]:
    addr = (addr or "").strip()
    if not addr:
        return "", "", ""
    toks = addr.split()
    city = ""; district = ""
    if toks:
        if toks[0] in METROS or toks[0] in PROVINCES:
            first = toks[0]
            city = CITY_SHORT.get(first, first.replace("도",""))
        elif re.search(r"(시|군|구)$", toks[0]):
            city = toks[0].replace("시","")
        for tk in toks[1:4]:
            if re.search(r"(구|군|시)$", tk):
                district = tk
                break
    region_phrase = f"{city} {district} 환자분들께".strip()
    region_phrase = re.sub(r"\s+", " ", region_phrase)
    return city, district, region_phrase


# ---------------- validators ----------------
def _valid_image_ref(ref: str) -> bool:
    return bool(re.fullmatch(r"(visit_images|therapy_images|result_images):\d+", ref or ""))

def ensure_plan_shape(plan: Dict[str, Any]) -> Dict[str, Any]:
    plan = _coerce_dict(plan or {})
    plan.setdefault("title", ""); plan.setdefault("summary", "")
    plan.setdefault("target_audience", ""); plan.setdefault("persona_focus", "")
    plan.setdefault("sections", []); plan.setdefault("keywords", [])
    plan.setdefault("call_to_action", "")
    plan.setdefault("geo_branding", {"clinic_alias":"", "region_line":""})
    plan.setdefault("meta_panel", {
        "address":"", "phone":"", "homepage":"", "map_link":"", "treatment_period":""
    })
    plan.setdefault("link_policy", {
        "homepage_in_body_once": True, "map_in_footer_only": True
    })
    # 섹션/이미지 강제 dict화
    norm_sections = []
    for sec in plan["sections"]:
        sec = _coerce_dict(sec)
        sec.setdefault("title",""); sec.setdefault("focus","")
        sec.setdefault("description",""); sec.setdefault("key_points",[])
        sec.setdefault("tone",""); sec.setdefault("images",[])
        sec.setdefault("keywords",[]); sec.setdefault("compliance_notes",[])
        sec.setdefault("evidence_hooks",[])
        norm_imgs = []
        for img in sec["images"]:
            img = _coerce_dict(img)
            img.setdefault("ref",""); img.setdefault("purpose",""); img.setdefault("alt","")
            norm_imgs.append(img)
        sec["images"] = norm_imgs
        norm_sections.append(sec)
    plan["sections"] = norm_sections
    return plan

def validate_plan(plan: Dict[str, Any]) -> (bool, List[str]):
    errs = []
    if not isinstance(plan, dict): return False, ["plan이 dict가 아님"]
    if not plan.get("title"): errs.append("title 누락")

    sections = plan.get("sections", [])
    if not isinstance(sections, list) or len(sections) < 5:
        errs.append("sections 5개 미만")
    else:
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                errs.append(f"sections[{i}] 형식 오류"); continue
            if not (sec.get("title") or "").strip():
                errs.append(f"sections[{i}].title 누락")
            if not (sec.get("description") or "").strip():
                errs.append(f"sections[{i}].description 누락")
            imgs = sec.get("images", [])
            if isinstance(imgs, list):
                for j, im in enumerate(imgs):
                    if not isinstance(im, dict):
                        errs.append(f"sections[{i}].images[{j}] 형식 오류"); 
                        continue
                    ref = im.get("ref", "")
                    if ref and not _valid_image_ref(ref):
                        errs.append(f"sections[{i}].images[{j}].ref 형식 오류: {ref}")

    return (len(errs) == 0), errs


# ---------------- prompts ----------------
SECTION_ORDER = [
    ("왜 이 글이 중요할까요?", "핵심 메시지 소개"),
    ("진단/검사 포인트", "내원 당시 상태와 검사"),
    ("치료는 이렇게 진행했어요", "치료 과정/재료/횟수"),
    ("치료 결과와 회복", "결과/예후/주의"),
    ("관리/예방 가이드", "관리·예방 팁"),
]

class PromptPack:
    def __init__(self):
        pref = Path("test_prompt")
        if not pref.exists():
            pref = Path("prompts")
        self.dir = pref
        self.batch_path = pref / "plan_candidate_batch_prompt.txt"
        self.single_path = pref / "plan_candidate_generation_prompt.txt"
        self.refine_path = pref / "plan_refine_to_five_sections.txt"

    def fill(self, template: str, mapping: Dict[str, Any]) -> str:
        out = template
        for k, v in mapping.items():
            if isinstance(v, (dict, list)):
                val = json.dumps(v, ensure_ascii=False)
            else:
                val = str(v)
            out = out.replace("{"+k+"}", val)
        return out

    def refine_template(self) -> str:
        if self.refine_path.exists():
            return read_text(self.refine_path)
        titles_roles = "\n".join([f"- {t} : {f}" for t, f in SECTION_ORDER])
        return (
            "당신은 치과 케이스 블로그 기획 전문가입니다.\n"
            "아래 input_data와 candidate_plan을 참고하여, '정확히 5개 섹션'으로 구성된 plan JSON을 생성하세요.\n"
            "각 섹션은 title, focus, description, key_points, tone, images, keywords, compliance_notes, evidence_hooks를 포함하고,"
            " description은 공백/한줄요약 금지, 구체적으로 작성합니다.\n"
            "질문 일부가 비어 있어도 임상 상식과 논리적 추론으로 자연스럽게 보완하세요.\n"
            f"섹션 순서/역할:\n{titles_roles}\n"
            "이미지가 있다면 visit_images:0 / therapy_images:0 / result_images:0를 적절히 배치(없으면 생략). "
            "출력은 아래 JSON 스키마 하나만:\n"
            "{\n"
            '  "title": "...",\n'
            '  "summary": "",\n'
            '  "target_audience": "",\n'
            '  "persona_focus": "",\n'
            '  "sections": [ { ...5 items... } ],\n'
            '  "keywords": [],\n'
            '  "call_to_action": "",\n'
            '  "geo_branding": {"clinic_alias":"", "region_line":""},\n'
            '  "meta_panel": {"address":"", "phone":"", "homepage":"", "map_link":"", "treatment_period":""},\n'
            '  "link_policy": {"homepage_in_body_once": true, "map_in_footer_only": true}\n'
            "}\n"
            "중괄호로 시작하는 하나의 JSON만 출력."
        )


# ---------------- Gemini ----------------
class GeminiClient:
    def __init__(self, model="gemini-1.5-pro", temperature=0.6, max_output_tokens=8192):
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


# ---------------- adherence (선택 점수) ----------------
def local_adherence(plan: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    score = 100
    secs = plan.get("sections", [])
    if not isinstance(secs, list) or len(secs) < 5:
        score -= 20
    for need in ["진단","치료","결과","관리"]:
        titles = " ".join([(s.get("title","") or "") for s in secs if isinstance(s, dict)])
        if need not in titles:
            score -= 5
    if not plan.get("persona_focus"):
        score -= 5
    return {
        "score_0_100": max(0, score),
        "reason_summary": "섹션5·요건부합 점수",
        "violations": [],
        "notes": []
    }


# ---------------- PlanAgent + Refiner ----------------
class PlanAgent:
    def __init__(self, model="gemini-1.5-pro"):
        self.prompts = PromptPack()
        self.gemini = GeminiClient(model=model)

    def _enrich_region(self, data: Dict[str, Any]) -> Dict[str, Any]:
        hospital = dict(data.get("hospital", {}))
        city = hospital.get("city",""); district = hospital.get("district",""); phrase = hospital.get("region_phrase","")
        if not city or not district or not phrase:
            c, d, p = extract_region_from_address(hospital.get("address",""))
            hospital.setdefault("city", c); hospital.setdefault("district", d); hospital.setdefault("region_phrase", p)
        data = dict(data); data["hospital"] = hospital
        return data

    def _vars_from_input(self, d: Dict[str, Any]) -> Dict[str, Any]:
        d = self._enrich_region(d)
        hospital = d.get("hospital", {})
        mapping = {
            "hospital_name": hospital.get("name",""),
            "hospital_address": hospital.get("address",""),
            "hospital_phone": hospital.get("phone",""),
            "hospital_homepage": hospital.get("homepage",""),
            "hospital_map_link": hospital.get("map_link",""),
            "hospital_save_name": hospital.get("save_name",""),
            "city": hospital.get("city",""),
            "district": hospital.get("district",""),
            "region_phrase": hospital.get("region_phrase",""),
            "category": d.get("category",""),
            "representative_persona": d.get("representative_persona",""),
            "persona_candidates": d.get("persona_candidates", []),
            "selected_personas": d.get("selected_personas", []),
            "question1_concept": d.get("question1_concept",""),
            "question2_condition": d.get("question2_condition",""),
            "question4_treatment": d.get("question4_treatment",""),
            "question6_result": d.get("question6_result",""),
            "question8_extra": d.get("question8_extra",""),
            "visit_images": d.get("visit_images", []),
            "therapy_images": d.get("therapy_images", []),
            "result_images": d.get("result_images", []),
            "images_index": d.get("images_index", {}),
            "link_policy_json": d.get("link_policy", {"homepage_in_body_once": True, "map_in_footer_only": True}),
            "geo_branding_hint": d.get("geo_branding", {}),
            "meta_panel_hint": d.get("meta_panel", {}),
        }
        return mapping

    def generate_candidates(self, input_data: Dict[str, Any], k: int) -> List[Dict[str, Any]]:
        vars_ = self._vars_from_input(input_data)

        # --- 배치 경로 ---
        if self.prompts.batch_path.exists():
            tmpl = read_text(self.prompts.batch_path)
            vars_["num_candidates"] = k
            prompt = self.prompts.fill(tmpl, vars_)
            raw = self.gemini.generate_text(prompt, temperature=0.7)
            try:
                obj = extract_json(raw)
                cands = obj.get("candidates", [])
                out = []
                for i, c in enumerate(cands, 1):
                    plan_raw = c.get("plan", {})
                    plan = ensure_plan_shape(_coerce_dict(plan_raw))
                    adh  = _coerce_dict(c.get("adherence", {}))
                    out.append({
                        "id": c.get("id", f"cand_{i}"),
                        "style": c.get("style",""),
                        "diversity": c.get("diversity",""),
                        "plan": plan,
                        "adherence": {
                            "score_0_100": adh.get("score_0_100"),
                            "reason_summary": adh.get("reason_summary",""),
                            "violations": adh.get("violations", []),
                            "notes": adh.get("notes", [])
                        }
                    })
                if out:
                    return out[:k]
            except Exception:
                pass  # 폴백

        # --- 단건 폴백 ---
        if not self.prompts.single_path.exists():
            raise FileNotFoundError("plan_candidate_generation_prompt.txt 없음")
        single_tmpl = read_text(self.prompts.single_path)
        styles = ["담백", "설명적", "대화형", "케이스저널", "체계적"]
        divers = ["질문형 도입", "명령형 소제목", "문장 짧게", "증거 선행", "체크리스트형 관리"]
        out = []
        for i in range(k):
            local = dict(vars_)
            local.update({
                "style_hint": styles[i % len(styles)],
                "diversity_guidance": divers[i % len(divers)],
                "candidate_seed": f"seed-{random.randint(1000,9999)}-#{i+1}"
            })
            prompt = self.prompts.fill(single_tmpl, local)
            raw = self.gemini.generate_text(prompt, temperature=0.6 + 0.1*(i % 3))
            plan_obj = extract_json(raw)  # 단건은 plan 자체 JSON을 반환하게 프롬프트 설계
            plan = ensure_plan_shape(_coerce_dict(plan_obj))
            out.append({
                "id": f"cand_{i+1}",
                "style": local["style_hint"],
                "diversity": local["diversity_guidance"],
                "plan": plan,
                "adherence": {}
            })
        return out

    def refine_to_five_sections(self, input_data: Dict[str, Any], candidate_plan: Dict[str, Any], strict: bool=False) -> Dict[str, Any]:
        """모델 리파인으로 정확히 5개 섹션 구성 + 비어있으면 문맥으로 보완"""
        vars_ = self._vars_from_input(input_data)
        tpl = self.prompts.refine_template()
        if strict:
            tpl += "\n\n추가 규칙: 모든 섹션의 description은 최소 3문장 이상, 구체적 사실/절차/결과 포함. 빈 문자열/한 문장/리스트 금지. 섹션 수는 꼭 5개."
        mapping = {
            **vars_,
            "candidate_plan": ensure_plan_shape(_coerce_dict(candidate_plan or {})),
            "section_order": [{"title": t, "focus": f} for t, f in SECTION_ORDER],
        }
        prompt = tpl.replace("{input_data}", json.dumps(vars_, ensure_ascii=False)) \
                    .replace("{candidate_plan}", json.dumps(mapping["candidate_plan"], ensure_ascii=False)) \
                    .replace("{section_order}", json.dumps(mapping["section_order"], ensure_ascii=False))
        raw = self.gemini.generate_text(prompt, temperature=0.5 if strict else 0.6)
        refined_obj = extract_json(raw)
        return ensure_plan_shape(_coerce_dict(refined_obj))


# ---------------- Save & Select ----------------
def local_adherence_or_use(plan_cand: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    adh = plan_cand.get("adherence", {}) or {}
    if isinstance(adh.get("score_0_100"), int):
        return adh
    return local_adherence(plan_cand.get("plan", {}), input_data)

def pick_best(cands: List[Dict[str, Any]], input_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    scored = []
    for c in cands:
        adh = local_adherence_or_use(c, input_data)
        scored.append((adh["score_0_100"], c, adh))
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_cand, best_adh = scored[0]
    reason = {
        "score_0_100": best_score,
        "reason_summary": best_adh.get("reason_summary","섹션5·요건부합 점수"),
        "violations": best_adh.get("violations", []),
        "notes": best_adh.get("notes", [])
    }
    return best_cand, reason


def save_final_plan(base_dir: Path, ts: str, mode: str, round_label: str, input_path: str,
                    selected_id: str, selection_reason: Dict[str, Any], plan: Dict[str, Any]) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    # ⬇️ 스왑: plan → *_plan_log.json / log → *_plan.json
    write_json(base_dir / f"{ts}_plan_log.json", plan)
    write_json(base_dir / f"{ts}_plan.json", {
        "mode": mode,
        "round": round_label,
        "input_path": input_path,
        "ts": ts,
        "selected_id": selected_id,
        "selection_reason": selection_reason
    })
    print(f"✅ 저장 완료: {base_dir/(ts+'_plan_log.json')} (plan)")
    print(f"✅ 저장 완료: {base_dir/(ts+'_plan.json')} (log)")



def generate_from_input_and_save(base_dir: Path, mode_label: str, round_label: str,
                                 input_data: Dict[str, Any], input_path: str, k: int) -> Optional[Dict[str, Any]]:
    agent = PlanAgent()
    candidates = agent.generate_candidates(input_data, k=k)
    if not candidates:
        print("⚠️ 후보 생성 실패"); return None
    best_cand, reason = pick_best(candidates, input_data)

    # 1차 리파인
    refined = agent.refine_to_five_sections(input_data, best_cand.get("plan", {}), strict=False)
    ok, errs = validate_plan(refined)

    # 2차 엄격 리파인(필요시)
    if not ok:
        refined = agent.refine_to_five_sections(input_data, refined, strict=True)
        ok, errs = validate_plan(refined)

    if not ok:
        print("⚠️ 최종안 검증 실패 → 저장하지 않습니다.")
        for e in errs: print(" -", e)
        return None

    ts = now_str()
    save_final_plan(base_dir, ts, mode_label, round_label, input_path, best_cand.get("id",""), reason, refined)
    return refined


# ---------- Orchestrations ----------
def run_test_generate() -> Optional[Dict[str, Any]]:
    """test: test_logs/test의 최신 *_input_log.json 기반. 없으면 1회 생성 여부 질문"""
    if InputAgent is None:
        raise RuntimeError("InputAgent 불러오기 실패. 프로젝트 루트에서 실행하세요.")

    base_dir = Path("test_logs/test")
    log_path = latest_input_log(base_dir)
    if log_path is None:
        print(f"⚠️ 최신 input 로그가 없습니다: {base_dir}/*_input_log.json")
        yn = input("지금 최신 input을 생성하시겠습니까? (Y/N): ").strip().lower()
        if yn != "y":
            print("생성하지 않습니다."); return None
        ia = InputAgent()
        data = ia.run_test()
        src_label = "(via InputAgent.run_test())"
    else:
        data = read_json(log_path)
        src_label = str(log_path)

    return generate_from_input_and_save(
        base_dir=base_dir,
        mode_label="test-generate",
        round_label="round2",
        input_data=data,
        input_path=src_label,
        k=4
    )

def run_use_generate() -> Optional[Dict[str, Any]]:
    """
    use 모드:
      - 1) 기존 선택(로그 기반) → test_logs/use 최신 *_input_log.json
      - 2) 직접 입력(새 수집)  → InputAgent.run_use()
    """
    if InputAgent is None:
        raise RuntimeError("InputAgent 불러오기 실패. 프로젝트 루트에서 실행하세요.")

    print("\nuse 데이터 소스를 선택하세요:")
    print("1) 기존 선택(로그 기반)  2) 직접 입력(새로 수집)")
    src = input("선택 (1/2): ").strip()

    base_dir = Path("test_logs/use")
    ia = InputAgent()

    if src == "1":
        log_path = latest_input_log(base_dir)
        if log_path is None:
            print(f"⚠️ 최신 input 로그가 없습니다: {base_dir}/*_input_log.json")
            yn = input("지금 직접 입력으로 새로 수집할까요? (Y/N): ").strip().lower()
            if yn != "y":
                print("생성하지 않습니다."); return None
            data = ia.run_use()
            src_label = "(via InputAgent.run_use())"
        else:
            data = read_json(log_path)
            src_label = str(log_path)
    elif src == "2":
        data = ia.run_use()
        src_label = "(via InputAgent.run_use())"
    else:
        print("⚠️ 잘못된 입력"); return None

    return generate_from_input_and_save(
        base_dir=base_dir,
        mode_label="use-generate",
        round_label="auto(3-5)",
        input_data=data,
        input_path=src_label,
        k=4
    )

def run_latest_plan_only(base_dir: Path) -> Optional[Dict[str, Any]]:
    # ⬇️ 스왑 반영: 최신 플랜은 *_plan_log.json에서 읽음
    latest_plan = latest_file_by_mtime(base_dir, "*_plan_log.json")
    if not latest_plan:
        print(f"⚠️ 최신 plan 결과가 없습니다: {base_dir}/*_plan_log.json")
        log_path = latest_file_by_mtime(base_dir, "*_plan.json")
        if log_path:
            print(f"ℹ️ 참고: 최신 로그 파일은 있습니다 → {log_path.name}")
        return None
    plan = read_json(latest_plan)
    print(f"📄 최신 PLAN(스왑): {latest_plan.name}")
    print("\n----- [PLAN SUMMARY] -----")
    print("title :", plan.get("title",""))
    print("secs  :", len(plan.get("sections", [])))
    print("cta   :", plan.get("call_to_action","")[:120])
    print("meta  :", plan.get("meta_panel",{}))
    return plan


def main():
    print("$ python agents/plan_agent.py")
    print("🔍 PlanAgent 시작\n")
    print("모드:")
    print("1) test         → 로그 기반 '바로 생성'(리파인 포함)")
    print("2) use          → ① 기존 선택(로그) ② 직접 입력(새 수집) (리파인 포함)")
    print("3) latest-view  → 최신 plan 요약 보기(폴더 선택)")
    sel = input("선택 (1/2/3): ").strip()
    try:
        if sel == "1":
            run_test_generate()
        elif sel == "2":
            run_use_generate()
        elif sel == "3":
            which = input("폴더 선택 (1: test_logs/test, 2: test_logs/use): ").strip()
            base = Path("test_logs/test") if which == "1" else Path("test_logs/use")
            run_latest_plan_only(base)
        else:
            print("⚠️ 잘못된 입력"); sys.exit(1)
    except Exception as e:
        print("❌ 오류:", e); sys.exit(1)

if __name__ == "__main__":
    main()
