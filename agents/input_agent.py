# input_agent.py
# -*- coding: utf-8 -*-

import json
import re
import sys
import difflib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

# ==================================
# Robust CSV loader (KR encodings)
# ==================================

def read_csv_kr(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc).fillna("")
        except Exception as e:
            last_err = e
            continue
    # 최후 수단
    try:
        return pd.read_csv(path, encoding="utf-8", errors="ignore").fillna("")
    except Exception:
        raise last_err


# ==================================
# CategoryDataIndex
# ==================================

class CategoryDataIndex:
    TOKEN_RE = re.compile(r"[^가-힣A-Za-z0-9\s]")

    def __init__(self, category_csv_path: str = "test_data/category_data.csv"):
        self.category_csv_path = Path(category_csv_path)
        self.tree: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        self._build_tree()

    def categories(self) -> List[str]:
        return sorted(list(self.tree.keys()))

    def symptoms_for(self, category: str) -> List[Tuple[str, str]]:
        cat = self.tree.get(category, {})
        return [(self._summarize_label(sym_txt), sym_txt) for sym_txt in sorted(cat.keys())]

    def procedures_for(self, category: str, symptom_text: str) -> List[Tuple[str, str]]:
        procs = self.tree.get(category, {}).get(symptom_text, {})
        return [(self._summarize_label(proc_txt), proc_txt) for proc_txt in sorted(procs.keys())]

    def treatments_for(self, category: str, symptom_text: str, procedure_text: str) -> List[Tuple[str, str]]:
        txs = self.tree.get(category, {}).get(symptom_text, {}).get(procedure_text, [])
        seen, out = set(), []
        for t in txs:
            if t not in seen:
                seen.add(t)
                out.append((self._summarize_label(t), t))
        return out

    def _build_tree(self) -> None:
        if not self.category_csv_path.exists():
            self.tree = {}
            return
        df = read_csv_kr(self.category_csv_path)
        for col in ["카테고리", "증상", "진료", "치료"]:
            if col not in df.columns:
                df[col] = ""
        df["카테고리"] = df["카테고리"].map(lambda x: str(x).strip())
        df["증상"] = df["증상"].map(lambda x: str(x).strip())
        df["진료"] = df["진료"].map(lambda x: str(x).strip())
        df["치료"] = df["치료"].map(lambda x: str(x).strip())

        tree: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        for _, row in df.iterrows():
            cat, sym, proc, tx = row["카테고리"], row["증상"], row["진료"], row["치료"]
            if not cat:
                continue
            tree.setdefault(cat, {})
            if sym:
                tree[cat].setdefault(sym, {})
                if proc:
                    tree[cat][sym].setdefault(proc, [])
                    if tx:
                        tree[cat][sym][proc].append(tx)
        self.tree = tree

    def _summarize_label(self, text: str, max_tokens: int = 6) -> str:
        if not isinstance(text, str) or not text.strip():
            return "(빈 내용)"
        t = self.TOKEN_RE.sub(" ", text)
        toks = [w for w in t.split() if len(w) >= 2]
        seen, out = set(), []
        for w in toks:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= max_tokens:
                break
        return " / ".join(out) if out else text[:20]


# ==================================
# InputAgent
# ==================================

class InputAgent:
    SUFFIXES = [
        "치과의원", "치과병원", "치과", "의원", "병원", "의료원", "메디컬센터", "센터", "클리닉", "덴탈"
    ]

    def __init__(
        self,
        test_data_path: str = "test_data/test_input_onlook.json",
        persona_csv_path: str = "test_data/persona_table.csv",
        category_csv_path: str = "test_data/category_data.csv",
        hospital_info_path: str = "test_data/test_hospital_info.json",
    ):
        self.test_data_path = Path(test_data_path)
        self.persona_df = read_csv_kr(persona_csv_path) if Path(persona_csv_path).exists() else pd.DataFrame()
        self.category_index = CategoryDataIndex(category_csv_path)
        self.valid_categories = sorted(self.persona_df["카테고리"].unique().tolist()) if (not self.persona_df.empty and "카테고리" in self.persona_df.columns) else []
        self.hospital_info_path = Path(hospital_info_path)

    # ---------- Normalization ----------
    def _normalize(self, s: str) -> str:
        s = str(s or "").strip().lower()
        s = re.sub(r"\(.*?\)", "", s)
        s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
        changed = True
        while changed and s:
            changed = False
            for suf in self.SUFFIXES:
                if s.endswith(suf):
                    s = s[: -len(suf)]
                    changed = True
        return s

    # ---------- Persona ----------
    def get_representative_personas(self, category: str) -> List[str]:
        if not category or self.persona_df.empty or "카테고리" not in self.persona_df.columns:
            return []
        row = self.persona_df[self.persona_df["카테고리"] == category]
        if row.empty:
            return []
        rep_raw = str(row.iloc[0].get("대표페르소나", "")).strip()
        out = [p.strip() for p in rep_raw.split(",") if p.strip()] if rep_raw else []
        return list(dict.fromkeys(out))

    def _pick_personas(self, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        print(f"\n선택 가능한 대표 페르소나: {candidates}")
        raw = input("사용할 페르소나를 쉼표로 입력 (엔터=모두): ").strip()
        if not raw:
            return list(dict.fromkeys(candidates))

        wanted = [x.strip().lower() for x in raw.split(",") if x.strip()]

        def aliases_for(p: str) -> List[str]:
            base = p.split("(")[0].strip().lower()
            inner = None
            if "(" in p and ")" in p:
                inner = p[p.find("(")+1:p.find(")")].strip().lower()
            al = [p.lower(), base]
            if inner:
                al.append(inner)
            return list(dict.fromkeys([a for a in al if a]))

        pairs = [(p, aliases_for(p)) for p in candidates]

        result = []
        for w in wanted:
            for original, aliases in pairs:
                if w in aliases and original not in result:
                    result.append(original)

        return list(dict.fromkeys(result or candidates))

    # ---------- Category guided selection ----------
    def _pick_from_options(self, title: str, options: List[Tuple[str, str]]) -> str:
        print(f"\n📋 {title}")
        if not options:
            return input("옵션이 없습니다. 직접 입력하세요: ").strip()
        for i, (label, value) in enumerate(options, 1):
            short = (value[:60] + "...") if len(value) > 60 else value
            print(f"{i}. {label}  |  {short}")
        while True:
            choice = input(f"선택 (1-{len(options)}) 또는 직접입력: ").strip()
            if not choice:
                print("⚠️ 입력이 비었습니다. 다시 선택해주세요.")
                continue
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
                print(f"⚠️ 1-{len(options)} 범위로 입력하세요.")
            except ValueError:
                return choice

    def _guided_selection(self, category: str) -> Dict[str, str]:
        sym = self._pick_from_options("[1단계] 증상 선택", self.category_index.symptoms_for(category))
        proc = self._pick_from_options("[2단계] 진료 선택", self.category_index.procedures_for(category, sym))
        tx = self._pick_from_options("[3단계] 치료 선택", self.category_index.treatments_for(category, sym, proc))
        return {"selected_symptom": sym, "selected_procedure": proc, "selected_treatment": tx}

    # ---------- Category input ----------
    def _input_category(self) -> str:
        all_cats = self.valid_categories or self.category_index.categories()
        print("\n📚 사용 가능한 카테고리:")
        print(", ".join(all_cats))
        while True:
            cat = input("카테고리를 입력하세요: ").strip()
            if not cat:
                print("⚠️ 빈 값입니다. 카테고리를 입력하세요.")
                continue
            if self.category_index.tree and cat not in self.category_index.tree:
                print("⚠️ category_data.csv에 해당 카테고리가 없습니다. 가능한 카테고리:", ", ".join(self.category_index.categories()))
                continue
            return cat

    # ---------- Hospital helpers ----------
    def _load_hospitals_list(self) -> List[Dict[str, str]]:
        if not self.hospital_info_path.exists():
            return []
        try:
            with open(self.hospital_info_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _ensure_url(self, u: str) -> str:
        u = (u or "").strip()
        if u and not re.match(r"^https?://", u, flags=re.I):
            u = "https://" + u
        return u

    def _derive_region(self, address: str) -> Tuple[str, str, str]:
        """
        address에서 city, district, region_phrase 추출(라이트 룰)
        예) '서울특별시 강남구 역삼동 ...' -> ('서울', '강남구', '서울 강남권')
            '경기도 화성시 동탄...' -> ('경기', '화성시', '경기 화성권')
        """
        addr = address or ""
        city = ""
        district = ""
        # city
        m = re.search(r"(서울|부산|대구|인천|광주|대전|울산)특별시|광역시", addr)
        if m:
            g = m.group(0)
            city = "서울" if "서울" in g else \
                   "부산" if "부산" in g else \
                   "대구" if "대구" in g else \
                   "인천" if "인천" in g else \
                   "광주" if "광주" in g else \
                   "대전" if "대전" in g else \
                   "울산" if "울산" in g else ""
        if not city:
            # 도/특별자치도 → 약칭
            m2 = re.search(r"(경기도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주특별자치도)", addr)
            if m2:
                dmap = {
                    "경기도": "경기", "강원도": "강원", "충청북도": "충북", "충청남도": "충남",
                    "전라북도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"
                }
                city = dmap.get(m2.group(1), "")
        # district (구/시/군)
        m3 = re.search(r"([가-힣]+구|[가-힣]+시|[가-힣]+군)", addr)
        if m3:
            district = m3.group(1)
        # region_phrase
        region_phrase = f"{city} {district}권".strip()
        region_phrase = region_phrase.replace("  ", " ").strip()
        return city, district, region_phrase

    def _build_geo_branding(self, hospital: Dict[str, str], city: str, district: str, region_phrase: str) -> Dict[str, str]:
        return {
            "clinic_alias": hospital.get("name", ""),
            "region_line": f"{city} {district} 환자분들께".strip()
        }

    def _build_meta_panel(self, hospital: Dict[str, str], period_hint: str = "") -> Dict[str, str]:
        return {
            "address": hospital.get("address", ""),
            "phone": hospital.get("phone", ""),
            "homepage": hospital.get("homepage", ""),
            "map_link": hospital.get("map_link", ""),
            "treatment_period": period_hint or ""
        }

    def _index_image_refs(self, visit_images: List[dict], therapy_images: List[dict], result_images: List[dict]) -> Dict[str, List[str]]:
        return {
            "visit_refs":  [f"visit_images:{i}"  for i, _ in enumerate(visit_images)],
            "therapy_refs":[f"therapy_images:{i}" for i, _ in enumerate(therapy_images)],
            "result_refs": [f"result_images:{i}" for i, _ in enumerate(result_images)],
        }

    def _extract_must_include_facts(self, text_blobs: List[str]) -> Dict[str, List[str]]:
        all_txt = " ".join([t for t in text_blobs if isinstance(t, str)])
        # FDI 번호(두 자리 또는 두 자리+하위) 라이트 추출
        fdi = sorted(set(re.findall(r"\b([1-4][1-8]|[1-4][0-8])\b", all_txt)))
        # 날짜/기간(YYYY.MM 또는 YYYY.MM.DD)
        dates = sorted(set(re.findall(r"\b(20\d{2}\.(?:0?[1-9]|1[0-2])(?:\.(?:0?[1-9]|[12]\d|3[01]))?)\b", all_txt)))
        # 회차/횟수 숫자 힌트
        counts = sorted(set(re.findall(r"\b(\d{1,2})회\b", all_txt)))
        # 장비/키워드
        equip_kw = []
        for kw in ["러버댐", "클램프", "CT", "파노라마", "근관확대", "Apex", "세척", "소독", "크라운", "임플란트"]:
            if kw in all_txt:
                equip_kw.append(kw)
        return {
            "tooth_fdi": fdi,
            "dates": dates,
            "counts": counts,
            "equip": equip_kw
        }

    # ---------- Images (filename + description) ----------
    def _input_image_pairs(self, prompt_title: str, save_name: str = "") -> List[Dict[str, str]]:
        print(f"\n🖼️ {prompt_title} — 이미지 파일명과 설명을 입력하세요.")
        pairs: List[Dict[str, str]] = []
        while True:
            more = input("추가하시겠습니까? (Y=추가 / Enter=그만): ").strip().lower()
            if more != "y":
                break
            filename = input(" - 파일명 (예: img001.png): ").strip()
            description = input(" - 설명 (예: 내원 시 파노라마): ").strip()

            if filename and save_name:
                expected_prefix = f"{save_name}_"
                if not filename.startswith(expected_prefix):
                    filename = f"{save_name}_{filename}"

            if filename:
                pairs.append({
                    "filename": filename,
                    "description": description
                })
        return pairs

    # ---------- Q1~Q8 (use-mode only) ----------
    def _ask_questions_8(self, save_name: str) -> Dict[str, object]:
        print("\n✍️ 8개 질문을 입력합니다.")
        q = {}
        q["question1_concept"] = input("Q1. 질환 개념/핵심 메시지: ").strip()
        q["question2_condition"] = input("Q2. 내원 당시 상태/검사(증상 중심): ").strip()
        q["visit_images"] = self._input_image_pairs("Q3. 내원 시 촬영 이미지", save_name=save_name)
        q["question4_treatment"] = input("Q4. 치료 내용(과정/재료/횟수 등): ").strip()
        q["therapy_images"] = self._input_image_pairs("Q5. 치료 중/후 이미지", save_name=save_name)
        q["question6_result"] = input("Q6. 치료 결과/예후/주의사항: ").strip()
        q["result_images"] = self._input_image_pairs("Q7. 결과 이미지", save_name=save_name)
        q["question8_extra"] = input("Q8. 기타 강조사항(통증/심미/기능 등): ").strip()
        return q

    # ---------- Q1~Q8 loader from log (test-mode) ----------
    def _load_q8_from_log(self, log_path: str) -> Dict[str, object]:
        keys = [
            "question1_concept", "question2_condition", "visit_images",
            "question4_treatment", "therapy_images", "question6_result",
            "result_images", "question8_extra"
        ]
        try:
            with open(log_path, encoding="utf-8") as f:
                data = json.load(f)
            result = {}
            for k in keys:
                if k.endswith("images"):
                    imgs = data.get(k, [])
                    if isinstance(imgs, list):
                        result[k] = [
                            {
                                "filename": (img.get("filename", "") if isinstance(img, dict) else ""),
                                "description": (img.get("description", "") if isinstance(img, dict) else "")
                            } for img in imgs
                        ]
                    else:
                        result[k] = []
                else:
                    result[k] = data.get(k, "")
            return result
        except Exception as e:
            print(f"⚠️ 로그 로드 실패: {e}")
            return {k: ([] if k.endswith("images") else "") for k in keys}

    # ---------- TEST ----------
    def run_test(self) -> Optional[dict]:
        hospital = self._get_hospital(allow_manual=False)
        save_name = hospital.get("save_name") or self._normalize(hospital.get("name", "")) or "hospital"

        if not self.test_data_path.exists():
            raise FileNotFoundError(f"테스트 입력 파일을 찾을 수 없습니다: {self.test_data_path}")
        with open(self.test_data_path, encoding="utf-8") as f:
            data = json.load(f)

        cases = [k for k in data.keys() if k.startswith("test_case_")]
        if not cases:
            print("❌ 사용 가능한 테스트 케이스가 없습니다.")
            return None

        print(f"\n📋 사용 가능한 테스트 케이스 ({len(cases)}개):")
        for k in cases:
            num = k.replace("test_case_", "")
            cat = data[k].get("category", "(미분류)")
            title = (data[k].get("question1_concept", "") or "").strip()
            if len(title) > 40:
                title = title[:40] + "..."
            print(f"{num}. [{cat}] {title}")

        while True:
            choice = input(f"\n케이스 선택 (1-{len(cases)}): ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(cases):
                    break
                else:
                    print(f"⚠️ 1-{len(cases)} 범위로 입력하세요.")
            except ValueError:
                print("⚠️ 숫자를 입력하세요.")

        selected = data[cases[idx]]

        category = selected.get("category", "").strip() or self._input_category()
        print(f"\n✅ 카테고리: {category}")

        picked = self._guided_selection(category)

        log_path = input("\n이전 입력 로그 파일 경로를 입력하세요 (없으면 엔터): ").strip()
        q8 = self._load_q8_from_log(log_path) if log_path else {
            k: ([] if k.endswith("images") else "") for k in
            ["question1_concept", "question2_condition", "visit_images",
             "question4_treatment", "therapy_images", "question6_result",
             "result_images", "question8_extra"]
        }

        persona_candidates = self.get_representative_personas(category)
        selected_personas = self._pick_personas(persona_candidates)

        persona_candidates = list(dict.fromkeys(persona_candidates))
        selected_personas = list(dict.fromkeys(selected_personas))
        representative_persona = selected_personas[0] if selected_personas else (persona_candidates[0] if persona_candidates else "")

        hospital = {**hospital, "save_name": save_name}

        # ----- 파생 필드 생성 -----
        city, district, region_phrase = self._derive_region(hospital.get("address", ""))
        geo_branding = self._build_geo_branding(hospital, city, district, region_phrase)
        meta_panel = self._build_meta_panel(hospital)
        link_policy = {"homepage_in_body_once": True, "map_in_footer_only": True}
        images_index = self._index_image_refs(q8.get("visit_images", []), q8.get("therapy_images", []), q8.get("result_images", []))
        content_flow_hint = "서론 → 진단 → 치료 → 결과 → 관리(FAQ)"
        # 페르소나 가이드
        rp = representative_persona
        if "심미" in rp: persona_structure_guide = "미적 문제→해결→변화→유지→심리효과"
        elif "통증" in rp: persona_structure_guide = "통증 원인→완화→치료→예방→일상개선"
        elif "기능" in rp: persona_structure_guide = "기능 문제→해결→회복→관리→일상개선"
        elif "잇몸" in rp: persona_structure_guide = "건강상태→위생→치료→검진→장기관리"
        else: persona_structure_guide = "문제→진단→치료→결과→관리"

        must_include_facts = self._extract_must_include_facts([
            q8.get("question1_concept",""),
            q8.get("question2_condition",""),
            q8.get("question4_treatment",""),
            q8.get("question6_result",""),
            q8.get("question8_extra","")
        ])

        res = {
            "mode": "test",
            "schema_version": "plan-input/1.0.0",
            "hospital": {**hospital, "city": city, "district": district, "region_phrase": region_phrase},
            "category": category,
            **picked,
            **q8,
            "persona_candidates": persona_candidates,
            "representative_persona": representative_persona,
            "selected_personas": selected_personas,
            # Plan-ready 추가
            "geo_branding": geo_branding,
            "meta_panel": meta_panel,
            "link_policy": link_policy,
            "images_index": images_index,
            "content_flow_hint": content_flow_hint,
            "persona_structure_guide": persona_structure_guide,
            "must_include_facts": must_include_facts,
        }
        self._validate_result(res)
        return res

    # ---------- USE ----------
    def run_use(self) -> dict:
        hospital = self._get_hospital(allow_manual=True)
        save_name = hospital.get("save_name") or self._normalize(hospital.get("name", "")) or "hospital"

        category = self._input_category()
        print(f"\n✅ 카테고리: {category}")

        picked = self._guided_selection(category)

        q8 = self._ask_questions_8(save_name=save_name)

        persona_candidates = self.get_representative_personas(category)
        selected_personas = self._pick_personas(persona_candidates)

        persona_candidates = list(dict.fromkeys(persona_candidates))
        selected_personas = list(dict.fromkeys(selected_personas))
        representative_persona = selected_personas[0] if selected_personas else (persona_candidates[0] if persona_candidates else "")

        hospital = {**hospital, "save_name": save_name}

        # ----- 파생 필드 생성 -----
        city, district, region_phrase = self._derive_region(hospital.get("address", ""))
        geo_branding = self._build_geo_branding(hospital, city, district, region_phrase)
        # Q8에서 치료기간 힌트가 들어왔다면 meta_panel에 반영 가능(라이트)
        period_hint = ""
        m_period = re.search(r"(20\d{2}\.\d{1,2}\.\d{1,2})\s*[-~]\s*(20\d{2}\.\d{1,2}\.\d{1,2})", q8.get("question6_result","") + " " + q8.get("question8_extra",""))
        if m_period:
            period_hint = f"{m_period.group(1)}–{m_period.group(2)}"
        meta_panel = self._build_meta_panel(hospital, period_hint=period_hint)
        link_policy = {"homepage_in_body_once": True, "map_in_footer_only": True}
        images_index = self._index_image_refs(q8.get("visit_images", []), q8.get("therapy_images", []), q8.get("result_images", []))
        content_flow_hint = "서론 → 진단 → 치료 → 결과 → 관리(FAQ)"
        rp = representative_persona
        if "심미" in rp: persona_structure_guide = "미적 문제→해결→변화→유지→심리효과"
        elif "통증" in rp: persona_structure_guide = "통증 원인→완화→치료→예방→일상개선"
        elif "기능" in rp: persona_structure_guide = "기능 문제→해결→회복→관리→일상개선"
        elif "잇몸" in rp: persona_structure_guide = "건강상태→위생→치료→검진→장기관리"
        else: persona_structure_guide = "문제→진단→치료→결과→관리"

        must_include_facts = self._extract_must_include_facts([
            q8.get("question1_concept",""),
            q8.get("question2_condition",""),
            q8.get("question4_treatment",""),
            q8.get("question6_result",""),
            q8.get("question8_extra","")
        ])

        res = {
            "mode": "use",
            "schema_version": "plan-input/1.0.0",
            "hospital": {**hospital, "city": city, "district": district, "region_phrase": region_phrase},
            "category": category,
            **picked,
            **q8,
            "persona_candidates": persona_candidates,
            "representative_persona": representative_persona,
            "selected_personas": selected_personas,
            # Plan-ready 추가
            "geo_branding": geo_branding,
            "meta_panel": meta_panel,
            "link_policy": link_policy,
            "images_index": images_index,
            "content_flow_hint": content_flow_hint,
            "persona_structure_guide": persona_structure_guide,
            "must_include_facts": must_include_facts,
        }
        self._validate_result(res)
        return res

    # ---------- Hospital selection/input ----------
    def _input_hospital_manual(self, prefill_name: str = "", hospitals: List[Dict[str, str]] = None) -> Dict[str, str]:
        hospitals = hospitals or []
        names = [h.get("name", "") for h in hospitals]
        norm_names = [self._normalize(n) for n in names]

        while True:
            print("\n[병원 정보 입력]")
            name = input(f"병원명 : ").strip() or prefill_name
            if not name:
                print("⚠️ 병원명은 필수입니다. 다시 입력해주세요.")
                continue

            q = self._normalize(name)
            if norm_names and q in norm_names:
                exist = hospitals[norm_names.index(q)]
                yn = input(f"❗ 이미 등록된 병원입니다: '{exist.get('name','')}'. 이 병원을 사용하시겠습니까? (Y/N): ").strip().lower()
                if yn == "y":
                    if not exist.get("save_name"):
                        exist = {**exist, "save_name": self._normalize(exist.get("name", ""))}
                    return exist
            else:
                close = difflib.get_close_matches(q, norm_names, n=1, cutoff=0.7) if norm_names else []
                if close:
                    cand = hospitals[norm_names.index(close[0])]
                    yn = input(f"❗ 비슷한 병원이 있습니다: '{cand.get('name','')}'. 이 병원을 사용하시겠습니까? (Y/N): ").strip().lower()
                    if yn == "y":
                        if not cand.get("save_name"):
                            cand = {**cand, "save_name": self._normalize(cand.get("name", ""))}
                        return cand

            save_name = input("저장용 병원명(save_name, 영문/숫자/소문자 권장, 미입력 시 자동 생성): ").strip()
            if not save_name:
                save_name = self._normalize(name)

            phone = input("전화번호: ").strip()
            address = input("주소: ").strip()
            homepage = self._ensure_url(input("홈페이지 URL: ").strip())
            map_link = self._ensure_url(input("지도 URL: ").strip())

            if not phone and not address:
                yn = input("ℹ️ 전화번호와 주소가 모두 비어 있습니다. 그대로 진행할까요? (Y=진행 / N=다시 입력): ").strip().lower()
                if yn != "y":
                    prefill_name = name
                    continue

            print("\n📌 입력 요약")
            print(f"- 병원명: {name}")
            print(f"- save_name: {save_name}")
            print(f"- 전화번호: {phone or '(비움)'}")
            print(f"- 주소: {address or '(비움)'}")
            print(f"- 홈페이지: {homepage or '(비움)'}")
            print(f"- 지도: {map_link or '(비움)'}")
            yn = input("이대로 등록할까요? (Y/N): ").strip().lower()
            if yn == "y":
                return {
                    "name": name,
                    "save_name": save_name,
                    "phone": phone,
                    "address": address,
                    "homepage": homepage,
                    "map_link": map_link
                }
            else:
                prefill_name = name
                print("다시 입력을 진행합니다.")

    def _load_hospitals_list(self) -> List[Dict[str, str]]:
        if not self.hospital_info_path.exists():
            return []
        try:
            with open(self.hospital_info_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _select_hospital(self, allow_manual: bool = True) -> Dict[str, str]:
        hospitals = self._load_hospitals_list()
        if not hospitals:
            if allow_manual:
                return self._input_hospital_manual()
            else:
                raise FileNotFoundError("hospital_info.json에 등록된 병원이 없습니다.")

        print("\n🏥 선택 가능한 병원:")
        for i, h in enumerate(hospitals, 1):
            addr = h.get("address", "")
            print(f"{i}. {h.get('name','')} ({addr})")

        names = [h.get("name", "") for h in hospitals]
        norm_names = [self._normalize(n) for n in names]

        while True:
            choice = input(f"병원 번호를 선택하세요 (1-{len(hospitals)}), 또는 병원명 직접 입력: ").strip()
            if not choice:
                print("⚠️ 입력이 비었습니다. 다시 시도해주세요.")
                continue

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(hospitals):
                    chosen = hospitals[idx]
                    if not chosen.get("save_name"):
                        chosen = {**chosen, "save_name": self._normalize(chosen.get("name", ""))}
                    return chosen
                print(f"⚠️ 1-{len(hospitals)} 범위로 입력하세요.")
                continue

            q = self._normalize(choice)

            if q in norm_names:
                chosen = hospitals[norm_names.index(q)]
                if not chosen.get("save_name"):
                    chosen = {**chosen, "save_name": self._normalize(chosen.get("name", ""))}
                return chosen

            partial_hits = [i for i, n in enumerate(norm_names) if q in n]
            if len(partial_hits) == 1:
                chosen = hospitals[partial_hits[0]]
                if not chosen.get("save_name"):
                    chosen = {**chosen, "save_name": self._normalize(chosen.get("name", ""))}
                return chosen
            elif len(partial_hits) > 1:
                print("\n🔎 여러 병원이 검색되었습니다. 아래에서 번호로 선택해주세요:")
                for j, i_hit in enumerate(partial_hits, 1):
                    h = hospitals[i_hit]
                    print(f"{j}. {h.get('name','')} ({h.get('address','')})")
                sub = input(f"선택 (1-{len(partial_hits)}): ").strip()
                if sub.isdigit():
                    sub_idx = int(sub) - 1
                    if 0 <= sub_idx < len(partial_hits):
                        chosen = hospitals[partial_hits[sub_idx]]
                        if not chosen.get("save_name"):
                            chosen = {**chosen, "save_name": self._normalize(chosen.get("name", ""))}
                        return chosen
                print("⚠️ 잘못된 입력입니다. 처음으로 돌아갑니다.")
                continue

            close = difflib.get_close_matches(q, norm_names, n=3, cutoff=0.6)
            if len(close) == 1:
                cand_idx = norm_names.index(close[0])
                cand = hospitals[cand_idx]
                yn = input(
                    f"❗ 해당하는 병원이 없습니다. 유사한 병원명을 제공드릴게요, 혹시 '{cand.get('name','')}'을(를) 입력하셨나요? (Y/N): "
                ).strip().lower()
                if yn == "y":
                    if not cand.get("save_name"):
                        cand = {**cand, "save_name": self._normalize(cand.get("name", ""))}
                    return cand
                else:
                    print("알겠습니다. 다시 입력해주세요.")
                    continue
            elif len(close) > 1:
                print("\n🧭 정확히 일치하지 않습니다. 유사한 병원 후보:")
                cand_idx = [norm_names.index(c) for c in close]
                for j, i_hit in enumerate(cand_idx, 1):
                    h = hospitals[i_hit]
                    print(f"{j}. {h.get('name','')} ({h.get('address','')})")
                sub = input(f"선택 (1-{len(cand_idx)}) 또는 Enter로 다시 입력: ").strip()
                if sub.isdigit():
                    sub_idx = int(sub) - 1
                    if 0 <= sub_idx < len(cand_idx):
                        chosen = hospitals[cand_idx[sub_idx]]
                        if not chosen.get("save_name"):
                            chosen = {**chosen, "save_name": self._normalize(chosen.get("name", ""))}
                        return chosen
                continue

            if allow_manual:
                yn = input("❗ 등록된 병원이 없습니다. 새 병원 정보를 입력하시겠습니까? (Y/N): ").strip().lower()
                if yn == "y":
                    return self._input_hospital_manual(prefill_name="", hospitals=hospitals)
                else:
                    print("목록에서 다시 선택하거나 병원명을 다시 입력해주세요.")
                    continue
            else:
                print("❗ 등록된 병원이 없습니다. 목록 중 번호를 선택하거나 정확한 이름을 입력해주세요.")
                continue

    def _get_hospital(self, allow_manual: bool) -> Dict[str, str]:
        return self._select_hospital(allow_manual=allow_manual)

    # ---------- Result validator ----------
    def _validate_result(self, res: dict) -> None:
        assert isinstance(res.get("persona_candidates", []), list)
        assert isinstance(res.get("selected_personas", []), list)
        for k in ["visit_images", "therapy_images", "result_images"]:
            assert isinstance(res.get(k, []), list)
            for img in res.get(k, []):
                assert isinstance(img, dict) and "filename" in img and "description" in img


# ==================================
# CLI (single-run)
# ==================================

if __name__ == "__main__":
    print("\n🔍 InputAgent 시작")
    print("test — 병원 → 테스트케이스 → 카테고리별 증상/진료/치료 → (질문8=로그 로드) → 페르소나")
    print("use  — 병원 → 카테고리 → 카테고리별 증상/진료/치료 → (질문8=직접 입력) → 페르소나")
    print("exit: 종료")

    agent = InputAgent()

    sel = input("\n모드 선택 (test, use, exit): ").strip().lower()
    if sel == "exit":
        sys.exit(0)
    elif sel == "test":
        res = agent.run_test()
        if res is None:
            print("❌ 사용자가 취소했습니다.")
            sys.exit(0)
    elif sel == "use":
        res = agent.run_use()
    else:
        print("⚠️ 잘못된 입력")
        sys.exit(1)

    # save log and exit
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("test_logs/use") if res.get("mode") == "use" else Path("test_logs/test")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{now}_input_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 로그 저장: {log_path}")

    print("\n" + "=" * 80)
    print("📋 [INPUT RESULT]")
    print("=" * 80)
    print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("=" * 80)

    sys.exit(0)
