# input_agent.py
# -*- coding: utf-8 -*-

"""
InputAgent (병원 입력/이미지 업로드 통합 + 임상 컨텍스트 빌더 + 유사 병원 제안 + S/P/T 선택)
- 공통 목표: test/use 모두 최종 스키마 동일 + case_id 업서트 + 날짜별 로그
- 질문 순서: Q1 → Q2 → Q3(이미지 배열) → Q4 → Q5(이미지 배열) → Q6 → Q7(이미지 배열) → Q8
- 이미지 필드: question3_visit_images / question5_therapy_images / question7_result_images
- 로그: test_logs/{mode}/{YYYYMMDD}/{YYYYMMDD_HHMMSS}_input_logs.json (배열 append)
"""

from __future__ import annotations

import os
import re
import json
import shutil
import pickle
import hashlib
import difflib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

import pandas as pd

# =========================
# 경로/시간 유틸 & JSON 헬퍼
# =========================
TEST_RESULT_PATH = Path("test_data/test_input_result.json")

def _today_str() -> str:
    return datetime.now().strftime("%Y%m%d")

def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _now_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_date_log_dir(mode: str) -> Path:
    d = Path(f"test_logs/{mode}/{_today_str()}")
    d.mkdir(parents=True, exist_ok=True)
    return d

def _append_json_array(path: Path, item: dict):
    arr = []
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                arr = json.load(f)
            if not isinstance(arr, list):
                arr = []
        except Exception:
            arr = []
    arr.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(arr, f, ensure_ascii=False, indent=2)

def _read_json(path: Path):
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _gen_case_id(save_name: str) -> str:
    ss = (save_name or "case").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"case_{ss}_{ts}"

# =========================
# CSV 로더 (인코딩 강인)
# =========================
def read_csv_kr(path: str | Path):
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
    try:
        return pd.read_csv(path, encoding="utf-8", errors="ignore").fillna("")
    except Exception:
        raise last_err

# =========================
# 레거시 케이스 → 새 스키마 변환기
# =========================
def migrate_legacy_case_to_new_schema(legacy_case: dict) -> dict:
    """
    legacy keys:
      - question3_visit_photo (str)
      - question5_therapy_photo (str, comma-separated)
      - question7_result_photo (str)
    to:
      - question3_visit_images: [{"filename":..., "description":""}, ...]
      - question5_therapy_images: [...]
      - question7_result_images: [...]
    pass-through: category, q1,q2,q4,q6,q8
    """
    out = {}
    for k in [
        "category",
        "question1_concept",
        "question2_condition",
        "question4_treatment",
        "question6_result",
        "question8_extra",
    ]:
        if k in legacy_case:
            out[k] = legacy_case.get(k, "")

    # Q3
    q3 = legacy_case.get("question3_visit_photo", "")
    if isinstance(q3, str) and q3.strip():
        out["question3_visit_images"] = [{"filename": q3.strip(), "description": ""}]
    else:
        out["question3_visit_images"] = []

    # Q5 (comma list)
    q5 = legacy_case.get("question5_therapy_photo", "")
    imgs5 = []
    if isinstance(q5, str) and q5.strip():
        for tok in [t.strip() for t in q5.split(",") if t.strip()]:
            imgs5.append({"filename": tok, "description": ""})
    out["question5_therapy_images"] = imgs5

    # Q7
    q7 = legacy_case.get("question7_result_photo", "")
    if isinstance(q7, str) and q7.strip():
        out["question7_result_images"] = [{"filename": q7.strip(), "description": ""}]
    else:
        out["question7_result_images"] = []

    out.setdefault("include_tooth_numbers", False)
    out.setdefault("tooth_numbers", [])
    return out

# =========================
# Clinical Context Builder
# =========================
class ClinicalContextBuilder:
    """
    category_data.csv(증상/진료/치료/카테고리) 기반
    - 카테고리별 키워드 KB
    - 행 인덱스(원문+토큰) 캐시
    - 스코어링/매칭/임상 흐름 빌더
    """

    NONWORD_RE = re.compile(r"[^가-힣A-Za-z0-9\s]")

    def __init__(self, category_csv_path: str, cache_dir: str = "cache"):
        self.category_csv_path = Path(category_csv_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.file_sig = self._file_signature(self.category_csv_path)
        self.category_kb = self._build_kb_from_csv()
        self.tree = self._load_or_build_tree_cache()

    @staticmethod
    def _file_signature(path: Path) -> str:
        st = os.stat(path)
        base = f"{path}|{st.st_mtime}|{st.st_size}"
        return hashlib.md5(base.encode()).hexdigest()

    @staticmethod
    def _clean_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @classmethod
    def _extract_keywords(cls, text: str, max_tokens: int = 40) -> List[str]:
        if not isinstance(text, str):
            return []
        text = cls.NONWORD_RE.sub(" ", text)
        toks = [t for t in text.split() if len(t) >= 2]
        return toks[:max_tokens]

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen, out = set(), []
        for x in items or []:
            x = str(x).strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def _build_kb_from_csv(self) -> Dict[str, Dict[str, List[str]]]:
        kb: Dict[str, Dict[str, List[str]]] = {}
        if not self.category_csv_path.exists():
            return kb
        df = read_csv_kr(self.category_csv_path)
        for col in ["증상", "진료", "치료", "카테고리"]:
            if col not in df.columns:
                df[col] = ""
        for _, row in df.iterrows():
            cat = self._clean_text(str(row.get("카테고리", "")))
            if not cat:
                continue
            kb.setdefault(cat, {"symptoms": [], "procedures": [], "treatments": []})
            kb[cat]["symptoms"] += self._extract_keywords(self._clean_text(row.get("증상", "")))
            kb[cat]["procedures"] += self._extract_keywords(self._clean_text(row.get("진료", "")))
            kb[cat]["treatments"] += self._extract_keywords(self._clean_text(row.get("치료", "")))
        for cat, d in kb.items():
            for f in ("symptoms", "procedures", "treatments"):
                d[f] = self._dedup_keep_order(d[f])
        return kb

    def _build_tree_index(self) -> Dict[str, List[Dict]]:
        tree: Dict[str, List[Dict]] = {}
        if not self.category_csv_path.exists():
            return tree
        df = read_csv_kr(self.category_csv_path)
        for col in ["증상", "진료", "치료", "카테고리"]:
            if col not in df.columns:
                df[col] = ""
        for _, row in df.iterrows():
            cat = self._clean_text(str(row.get("카테고리", "")))
            if not cat:
                continue
            sym_txt = self._clean_text(row.get("증상", ""))
            prc_txt = self._clean_text(row.get("진료", ""))
            tx_txt = self._clean_text(row.get("치료", ""))
            entry = {
                "symptom_text": sym_txt,
                "procedure_text": prc_txt,
                "treatment_text": tx_txt,
                "sym_tokens": self._extract_keywords(sym_txt),
                "proc_tokens": self._extract_keywords(prc_txt),
                "tx_tokens": self._extract_keywords(tx_txt),
            }
            tree.setdefault(cat, []).append(entry)
        return tree

    def _load_or_build_tree_cache(self) -> Dict[str, List[Dict]]:
        cache_path = self.cache_dir / f"{self.file_sig}.tree.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        tree = self._build_tree_index()
        with open(cache_path, "wb") as f:
            pickle.dump(tree, f)
        return tree

    def normalize(self, symptoms=None, procedures=None, treatments=None, fdi_teeth=None):
        def _norm_list(items: List[str]) -> List[str]:
            clean = [self._clean_text(x) for x in (items or []) if self._clean_text(x)]
            return self._dedup_keep_order(clean)
        return {
            "symptoms": _norm_list(symptoms or []),
            "procedures": _norm_list(procedures or []),
            "treatments": _norm_list(treatments or []),
            "tooth_numbers": self._dedup_keep_order(fdi_teeth or []),
        }

    def score_categories(self, normalized: Dict, category_hint: Optional[str] = None) -> Dict[str, float]:
        weights = {"symptoms": 1.0, "procedures": 0.8, "treatments": 1.2}
        scores = {cat: 0.0 for cat in self.category_kb.keys()}
        for cat, keys in self.category_kb.items():
            s = 0.0
            for field, terms in keys.items():
                base = normalized.get(field, [])
                for b in base:
                    if b in terms:
                        s += 1.0 * weights.get(field, 1.0)
                    else:
                        if any((b in t) or (t in b) for t in terms if len(t) >= 2):
                            s += 0.5 * weights.get(field, 1.0)
            scores[cat] = s
        if category_hint and category_hint in scores:
            scores[category_hint] *= 1.15
        maxv = max(scores.values()) if scores else 1.0
        if maxv > 0:
            scores = {k: round(v / maxv, 4) for k, v in scores.items()}
        return scores

    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 0.0
        inter = len(sa & sb)
        union = len(sa | sb) or 1
        return inter / union

    def _row_similarity(self, norm: Dict, entry: Dict) -> float:
        w_sym, w_prc, w_tx = 1.0, 0.8, 1.2
        s1 = self._jaccard(norm.get("symptoms", []), entry["sym_tokens"])
        s2 = self._jaccard(norm.get("procedures", []), entry["proc_tokens"])
        s3 = self._jaccard(norm.get("treatments", []), entry["tx_tokens"])
        return w_sym * s1 + w_prc * s2 + w_tx * s3

    def match_topk(self, normalized: Dict, primary_cat: str, topk: int = 5) -> Dict:
        candidates = self.tree.get(primary_cat, [])
        if not candidates:
            return {"matches": [], "treatments": []}
        scored: List[Tuple[float, Dict]] = []
        for e in candidates:
            scored.append((self._row_similarity(normalized, e), e))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:topk]
        tx_scores: Dict[str, float] = {}
        matches = []
        for score, e in top:
            tx_text = e["treatment_text"] or ""
            matches.append(
                {
                    "score": round(score, 4),
                    "symptom_text": e["symptom_text"],
                    "procedure_text": e["procedure_text"],
                    "treatment_text": tx_text,
                }
            )
            if tx_text:
                tx_scores[tx_text] = tx_scores.get(tx_text, 0.0) + score
        treatments = [{"name": k, "score": round(v, 4)} for k, v in sorted(tx_scores.items(), key=lambda x: x[1], reverse=True)]
        return {"matches": matches, "treatments": treatments}

    def build_flow(self, normalized: Dict, scores: Dict[str, float], notes: Optional[str] = None, visit_purpose: Optional[str] = None):
        primary = max(scores, key=scores.get) if scores else ""
        secondary = sorted([k for k in scores if k != primary], key=lambda k: scores[k], reverse=True)[:2]
        flow = []
        sym = ", ".join(normalized.get("symptoms", []) or ["(무)"])
        flow.append(f"주호소: {sym}" + (f" / 목적: {visit_purpose}" if visit_purpose else ""))
        proc = ", ".join(normalized.get("procedures", []) or [])
        if proc:
            flow.append(f"검사: {proc}")
        tx = ", ".join(normalized.get("treatments", []) or ["(미정)"])
        flow.append(f"치료계획: {tx}")
        flow.append("예후/관리: 정기 검진 및 위생관리 안내")
        if notes:
            flow.append(f"메모: {notes}")
        return {"primary_category": primary, "secondary_categories": secondary, "flow": flow}

    def build(self, raw: Dict, topk: int = 5) -> Dict:
        q1 = str(raw.get("question1_concept", ""))
        q2 = str(raw.get("question2_condition", ""))
        q4 = str(raw.get("question4_treatment", ""))
        q6 = str(raw.get("question6_result", ""))
        q8 = str(raw.get("question8_extra", ""))

        symptoms = self._extract_keywords(q2 + " " + q1)
        procedures = self._extract_keywords(q2)
        treatments = self._extract_keywords(q4 + " " + q6 + " " + q8)

        use_fdi = bool(raw.get("include_tooth_numbers", False))
        tooth_numbers = raw.get("tooth_numbers", []) if use_fdi else []

        normalized = self.normalize(
            symptoms=symptoms,
            procedures=procedures,
            treatments=treatments,
            fdi_teeth=tooth_numbers,
        )
        scores = self.score_categories(normalized, raw.get("category"))
        flow = self.build_flow(normalized, scores, notes=None, visit_purpose=None)

        primary_cat = flow.get("primary_category", "")
        topk_pack = self.match_topk(normalized, primary_cat, topk=topk) if primary_cat else {"matches": [], "treatments": []}

        return {
            "normalized": normalized,
            "category_scores": scores,
            **flow,
            "top_matches": topk_pack["matches"],
            "recommended_treatments": topk_pack["treatments"],
        }

# ==============
# InputAgent
# ==============
class InputAgent:
    def __init__(
        self,
        input_data: Optional[dict] = None,
        case_num: str = "1",
        test_data_path: str = "test_data/test_input_onlook.json",
        persona_csv_path: str = "test_data/persona_table.csv",
        hospital_info_path: str = "test_data/test_hospital_info.json",
        hospital_image_path: str = "test_data/hospital_image",
        category_csv_path: str = "test_data/category_data.csv",
        select_csv_path: str = "test_data/select_data.csv",
        cache_dir: str = "cache",
    ):
        self.case_num = case_num
        self.test_data_path = Path(test_data_path)
        self.input_data = input_data

        self.persona_df = read_csv_kr(persona_csv_path)
        self.valid_categories = self.persona_df["카테고리"].unique().tolist()

        self.hospital_info_path = Path(hospital_info_path)
        self.hospital_image_path = Path(hospital_image_path)

        self.hospital_list: List[Dict[str, Any]] = []
        if self.hospital_info_path.exists():
            try:
                with open(self.hospital_info_path, encoding="utf-8") as f:
                    self.hospital_list = json.load(f) or []
            except Exception:
                self.hospital_list = []

        self.select_df = None
        self.select_csv_path = Path(select_csv_path)
        if self.select_csv_path.exists():
            self.select_df = read_csv_kr(self.select_csv_path)
            for col in ["카테고리", "증상_선택", "진료_선택", "치료_선택"]:
                if col not in self.select_df.columns:
                    self.select_df[col] = ""

        self.context_builder = ClinicalContextBuilder(category_csv_path, cache_dir=cache_dir)

    # ---------- 업서트 & 로그 ----------
    def upsert_test_input_result(self, payload: dict) -> None:
        db = _read_json(TEST_RESULT_PATH)
        if not isinstance(db, dict):
            db = {}
        case_id = payload.get("case_id")
        if not case_id:
            case_id = _gen_case_id(payload.get("hospital", {}).get("save_name", "case"))
            payload["case_id"] = case_id
        db[case_id] = payload
        _write_json(TEST_RESULT_PATH, db)
        print(f"✅ 업서트 완료 → {TEST_RESULT_PATH.name}  (key={case_id})")

    def save_log(self, result: dict, mode: str = "use") -> None:
        date_dir = _ensure_date_log_dir(mode)
        # 파일명: 오늘날짜_타임스탬프_input_logs.json  (예: 20250812_114416_input_logs.json)
        filename = f"{_now_compact()}_input_logs.json"
        log_path = date_dir / filename

        log_item = dict(result)
        log_item["timestamp"] = _now_str()
        log_item["mode"] = mode

        _append_json_array(log_path, log_item)
        print(f"📝 로그 저장 → {log_path}")

    def _finalize_and_save(self, data: dict, mode: str) -> dict:
        # ensure prefixed image keys
        if "question3_visit_images" not in data:
            data["question3_visit_images"] = data.pop("visit_images", [])
        if "question5_therapy_images" not in data:
            data["question5_therapy_images"] = data.pop("therapy_images", [])
        if "question7_result_images" not in data:
            data["question7_result_images"] = data.pop("result_images", [])

        # case_id
        if not data.get("case_id"):
            save_name = (data.get("hospital") or {}).get("save_name", "")
            data["case_id"] = _gen_case_id(save_name)

        # 업서트 여부
        yn = input("저장/업데이트 하시겠습니까? (Y=등록/업데이트, N=로그만): ").strip().lower()
        if yn == "y":
            self.upsert_test_input_result(data)
        else:
            print("ℹ️ 업서트 생략, 로그만 저장합니다.")

        # 로그
        self.save_log(data, mode=mode)
        return data

    # ---------- 병원 유사도 ----------
    @staticmethod
    def _norm_name(s: str) -> str:
        s = (s or "").strip().lower()
        s = re.sub(r"(치과|의원|병원)$", "", s)
        s = re.sub(r"[^가-힣a-z0-9]+", "", s)
        return s

    def _similarity(self, a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, self._norm_name(a), self._norm_name(b)).ratio()

    def suggest_hospitals(self, query: str, topn: int = 5, threshold: float = 0.55) -> List[Tuple[float, Dict]]:
        if not query or not self.hospital_list:
            return []
        scored = []
        for h in self.hospital_list:
            name = h.get("name", "")
            save = h.get("save_name", "")
            score = max(self._similarity(query, name), self._similarity(query, save))
            if score >= threshold:
                scored.append((score, h))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:topn]

    # ---------- 병원 이미지 파일 찾기 ----------
    def find_image_file(self, name: str, keyword: str) -> Optional[str]:
        for ext in ["png", "jpg", "jpeg", "webp"]:
            for file in self.hospital_image_path.glob(f"{name}_*{keyword}.{ext}"):
                return file.name
        return None

    # ---------- 병원 정보: 정확 일치 ----------
    def load_hospital_info_exact(self, name: str) -> Optional[dict]:
        if not self.hospital_list:
            return None
        for h in self.hospital_list:
            if h.get("name") == name or h.get("save_name") == name:
                save_name = h.get("save_name", name)
                h = dict(h)
                h["logo"] = self.find_image_file(save_name, "_logo")
                h["business_card"] = self.find_image_file(save_name, "_business_card")
                return h
        return None

    # ---------- 병원 정보: 수동 입력 ----------
    def manual_input_hospital_info(self, name: Optional[str] = None) -> dict:
        print("\n[병원 정보 수동 입력 시작]")
        name = name or input("병원 이름을 입력하세요: ").strip()
        save_name = input("병원 저장명을 입력하세요 (예: hani): ").strip()
        homepage = input("홈페이지 URL을 입력하세요: ").strip()
        phone = input("전화번호를 입력하세요: ").strip()
        address = input("병원 주소를 입력하세요 (예: 서울특별시 강남구 논현동 123): ").strip()
        map_link = input("네이버 지도 URL을 입력하세요 (없으면 Enter): ").strip() or None

        print("\n[병원 이미지 매핑]")
        logo_file = input("로고 이미지 파일명을 입력하세요 (예: logo1.png): ").strip()
        card_file = input("명함 이미지 파일명을 입력하세요 (예: card1.jpg): ").strip()

        mapping = {}
        if logo_file:
            mapping[logo_file] = f"{save_name}_logo"
        if card_file:
            mapping[card_file] = f"{save_name}_business_card"
        self.process_uploaded_images(mapping)

        logo = self.find_image_file(save_name, "_logo")
        business_card = self.find_image_file(save_name, "_business_card")

        # 수동 입력을 병원 DB에 저장할지 여부 (옵션)
        yn = input("이 병원 정보를 병원 목록에 저장하시겠습니까? (Y/N): ").strip().lower()
        if yn == "y":
            self._upsert_hospital_list({
                "name": name,
                "save_name": save_name,
                "homepage": homepage,
                "phone": phone,
                "address": address,
                "map_link": map_link
            })

        return {
            "name": name,
            "save_name": save_name,
            "homepage": homepage,
            "phone": phone,
            "address": address,
            "map_link": map_link,
            "logo": logo,
            "business_card": business_card,
        }

    def _upsert_hospital_list(self, hospital: dict):
        # name/save_name 기준 업서트
        key = (hospital.get("save_name") or hospital.get("name") or "").strip()
        if not key:
            return
        existed = False
        for i, h in enumerate(self.hospital_list):
            if h.get("save_name") == hospital.get("save_name") or h.get("name") == hospital.get("name"):
                self.hospital_list[i] = {**h, **hospital}
                existed = True
                break
        if not existed:
            self.hospital_list.append(hospital)
        try:
            with open(self.hospital_info_path, "w", encoding="utf-8") as f:
                json.dump(self.hospital_list, f, ensure_ascii=False, indent=2)
            print(f"✅ 병원 정보 업서트 완료 → {self.hospital_info_path.name}")
        except Exception as e:
            print(f"⚠️ 병원 정보 저장 실패: {e}")

    # ---------- 병원 정보: 유사 제안 포함 인터랙티브 ----------
    def interactive_select_hospital(self, typed_name: str) -> dict:
        exact = self.load_hospital_info_exact(typed_name)
        if exact:
            return exact

        suggestions = self.suggest_hospitals(typed_name, topn=5, threshold=0.55)
        if suggestions:
            print("\n🔎 입력하신 이름과 유사한 병원이 있습니다:")
            for i, (score, h) in enumerate(suggestions, 1):
                addr = h.get("address", "")
                print(f"  {i}. {h.get('name','(이름없음)')}  | score={score:.2f}  | {addr}")
            print("  R. 다시 입력")
            print("  M. 수동 입력으로 진행")
            sel = input("번호를 선택하세요 (Enter=다시 입력): ").strip().lower()
            if sel == "m":
                return self.manual_input_hospital_info(typed_name)
            if sel == "r" or sel == "":
                new_name = input("다시 병원 이름을 입력하세요: ").strip()
                return self.interactive_select_hospital(new_name)
            if sel.isdigit():
                idx = int(sel) - 1
                if 0 <= idx < len(suggestions):
                    chosen = dict(suggestions[idx][1])
                    save_name = chosen.get("save_name", chosen.get("name", ""))
                    chosen["logo"] = self.find_image_file(save_name, "_logo")
                    chosen["business_card"] = self.find_image_file(save_name, "_business_card")
                    print(f"✅ 선택됨: {chosen.get('name')} ({chosen.get('address','')})")
                    return chosen
            print("⚠️ 잘못된 선택입니다. 수동 입력으로 전환합니다.")
            return self.manual_input_hospital_info(typed_name)
        else:
            print("\nℹ️ 유사한 병원을 찾지 못했습니다. 수동 입력으로 진행합니다.")
            return self.manual_input_hospital_info(typed_name)

    # ---------- 업로드된(또는 테스트) 이미지 복사/정규화 ----------
    def process_uploaded_images(
        self,
        mapping: dict,
        test_image_dir: Path = Path("test_data/test_image"),
        hospital_image_dir: Path = Path("test_data/hospital_image"),
    ) -> None:
        """
        원본: test_data/test_image/원본파일
        대상: test_data/hospital_image/{save_name}_{원파일명(정규화)}_{logo|business_card}.ext
        """
        hospital_image_dir.mkdir(parents=True, exist_ok=True)
        for original_filename, mapped_stem in mapping.items():
            original_path = test_image_dir / original_filename
            if not original_path.exists():
                print(f"❌ 파일 없음: {original_filename}")
                continue
            base_stem = original_path.stem
            safe_base = re.sub(r"[^가-힣A-Za-z0-9_-]+", "_", base_stem).strip("_")
            suffix = "_logo" if mapped_stem.endswith("_logo") else "_business_card"
            save_name = mapped_stem.split("_")[0]
            ext = original_path.suffix.lower()
            new_filename = f"{save_name}_{safe_base}{suffix}{ext}"
            new_path = hospital_image_dir / new_filename
            try:
                shutil.copy(original_path, new_path)
                print(f"✅ {original_filename} → {new_filename} 복사 완료")
            except Exception as e:
                print(f"⚠️ 복사 실패: {original_filename} → {new_filename} | {e}")

    # ---------- 주소에서 지역 추출 ----------
    def extract_region_info(self, address: str) -> dict:
        parts = (address or "").split()
        if len(parts) < 2:
            return {"city": "", "district": "", "region_phrase": ""}
        city = parts[0].replace("특별시", "").replace("광역시", "").replace("자치시", "").replace("도", "")
        district = parts[1].replace("시", "").replace("군", "").replace("구", "")
        return {"city": city, "district": district, "region_phrase": f"{city} {district}"}

    # ---------- 선택 유틸 (목록에서 번호로 선택) ----------
    @staticmethod
    def _choose_from_list(title: str, options: List[str]) -> str:
        if not options:
            print(f"⚠️ {title} 옵션이 없습니다.")
            return ""
        print(f"\n🔽 {title} 선택:")
        uniq, seen = [], set()
        for o in options:
            o = str(o).strip()
            if o and o not in seen:
                uniq.append(o)
                seen.add(o)
        for i, opt in enumerate(uniq, 1):
            short = opt if len(opt) <= 60 else (opt[:58] + "…")
            print(f"  {i}. {short}")
        while True:
            sel = input(f"번호를 선택하세요 (1~{len(uniq)}): ").strip()
            if sel.isdigit() and 1 <= int(sel) <= len(uniq):
                return uniq[int(sel) - 1]
            print("잘못된 입력입니다. 다시 선택하세요.")

    # ---------- S/P/T 선택 ----------
    def select_spt(self, category: str) -> Dict[str, str]:
        if self.select_df is None or not category:
            return {"selected_symptom": "", "selected_procedure": "", "selected_treatment": ""}
        df_cat = self.select_df[self.select_df["카테고리"] == category]
        if df_cat.empty:
            print("⚠️ 선택한 카테고리에 해당하는 항목이 없습니다.")
            return {"selected_symptom": "", "selected_procedure": "", "selected_treatment": ""}
        symptom = self._choose_from_list("증상", df_cat["증상_선택"].tolist())
        df_sym = df_cat[df_cat["증상_선택"] == symptom]
        procedure = self._choose_from_list("진료", df_sym["진료_선택"].tolist())
        df_proc = df_sym[df_sym["진료_선택"] == procedure]
        treatment = self._choose_from_list("치료", df_proc["치료_선택"].tolist())
        return {
            "selected_symptom": symptom,
            "selected_procedure": procedure,
            "selected_treatment": treatment,
        }

    # ---------- 이미지 입력(Q3/Q5/Q7: 배열) ----------
    def _find_source_image(self, filename: str, search_dirs: Optional[List[Path]] = None) -> Optional[Path]:
        if not filename:
            return None
        filename = Path(filename).name
        search_dirs = search_dirs or [Path("test_data/test_image"), Path("images"), Path(".")]
        hits: List[Path] = []
        for root in search_dirs:
            if not Path(root).exists():
                continue
            for p in Path(root).rglob("*"):
                if p.is_file() and p.name.lower() == filename.lower():
                    hits.append(p)
        if not hits:
            return None
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0]

    def _normalize_and_copy_image(self, filename: str, save_name: str, dest_dir: Path = Path("test_data/test_image"), suffix: str = "") -> str:
        src = self._find_source_image(filename)
        base, ext = os.path.splitext(Path(filename).name)
        safe_base = re.sub(r"[^가-힣A-Za-z0-9_-]+", "_", base).strip("_")
        target_name = f"{save_name}_{safe_base}{suffix}{ext}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / target_name
        if not src:
            print(f"⚠️ 소스 이미지가 존재하지 않습니다: {filename}")
            return target_name
        try:
            shutil.copy2(src, dst)
            print(f"✅ 이미지 복사: {src} → {dst}")
        except Exception as e:
            print(f"❌ 이미지 복사 실패: {e}")
        return target_name

    def _input_image_pairs(self, prompt_title: str, save_name: str = "") -> List[Dict[str, str]]:
        print(f"\n🖼️ {prompt_title} — 이미지 파일명과 설명을 입력해 주세요.")
        print("   (예: 파일명: '바나나.jpg'  ·  설명: '초진 파노라마')")
        pairs: List[Dict[str, str]] = []
        while True:
            more = input("추가하시겠습니까? (Y=추가 / Enter=그만): ").strip().lower()
            if more != "y":
                break
            filename = input(" - 파일명/경로: ").strip()
            description = input(" - 설명: ").strip()
            if not filename:
                print("⚠️ 파일명이 비었습니다. 건너뜁니다.")
                continue
            normalized_basename = self._normalize_and_copy_image(
                filename=filename, save_name=save_name, dest_dir=Path("test_data/test_image"), suffix=""
            )
            pairs.append({"filename": normalized_basename, "description": description})
        return pairs

    # ---------- FDI 치식번호 입력 ----------
    @staticmethod
    def _input_tooth_numbers(prompt: str) -> List[str]:
        valid_pat = re.compile(r"^(?:1[1-8]|2[1-8]|3[1-8]|4[1-8])$")  # 11~18,21~28,31~38,41~48
        while True:
            raw = input(prompt).strip()
            if raw == "":
                return []  # 빈 입력 허용
            items = [x.strip() for x in raw.split(",") if x.strip()]
            invalid = [it for it in items if not valid_pat.fullmatch(it)]
            if invalid:
                print(f"❌ 유효하지 않은 치식번호: {', '.join(invalid)}")
                print("   → 허용: 11~18, 21~28, 31~38, 41~48 (예: 11, 21, 36)")
                continue
            return list(dict.fromkeys(items))

    # ---------- 페르소나 선택(별칭 허용) ----------
    def get_representative_personas(self, category: str) -> List[str]:
        if not category:
            return []
        row = self.persona_df[self.persona_df["카테고리"] == category]
        if row.empty:
            return []
        rep_raw = str(row.iloc[0].get("대표페르소나", "")).strip()
        return [p.strip() for p in rep_raw.split(",") if p.strip()] if rep_raw else []

    def select_personas(self, available_personas: List[str]) -> List[str]:
        if not available_personas:
            return []
        alias: Dict[str, str] = {}
        for full in available_personas:
            f = (full or "").strip()
            if not f:
                continue
            base = f.split("(")[0].strip()
            code = ""
            m = re.search(r"\(([^)]+)\)", f)
            if m:
                code = m.group(1).strip()
            candidates = {f, base}
            if code:
                candidates.add(code)
            more = set()
            for c in list(candidates):
                more.add(c.lower())
                more.add(c.upper())
                more.add(re.sub(r"\s+", "", c))
                more.add(re.sub(r"\s+", "", c).lower())
                more.add(re.sub(r"\s+", "", c).upper())
            candidates |= more
            for k in candidates:
                key = k.strip().lower()
                if key and key not in alias:
                    alias[key] = f
        while True:
            print(f"선택 가능한 대표 페르소나: {available_personas}")
            raw = input("사용할 페르소나를 쉼표로 입력하거나, 엔터=모두: ").strip()
            if raw == "":
                return available_personas
            tokens = [t.strip() for t in raw.split(",") if t.strip()]
            chosen, invalid, seen = [], [], set()
            for t in tokens:
                k = t.strip().lower()
                k2 = re.sub(r"\s+", "", k)
                hit = alias.get(k) or alias.get(k2) or alias.get(k.upper().lower()) or alias.get(k2.upper().lower())
                if hit:
                    if hit not in seen:
                        chosen.append(hit)
                        seen.add(hit)
                else:
                    invalid.append(t)
            if not invalid and chosen:
                return list(dict.fromkeys(chosen))
            print(f"잘못된 페르소나: {invalid} — 다시 선택")

    # ---------- Clinical context wrapper ----------
    def _build_clinical_context(self, raw: Dict) -> Dict:
        ctx = self.context_builder.build(
            {
                "question1_concept": raw.get("question1_concept", ""),
                "question2_condition": raw.get("question2_condition", ""),
                "question4_treatment": raw.get("question4_treatment", ""),
                "question6_result": raw.get("question6_result", ""),
                "question8_extra": raw.get("question8_extra", ""),
                "include_tooth_numbers": raw.get("include_tooth_numbers", False),
                "tooth_numbers": raw.get("tooth_numbers", []),
                "category": raw.get("category"),
            },
            topk=5,
        )
        return ctx

    # ---------- 수동 전체 입력 (질문 순서 Q1→…→Q8) ----------
    def _manual_questions_q1_to_q8(self, save_name: str) -> Dict[str, Any]:
        q1 = input("Q1. 질환 개념 및 강조 메시지: ").strip()
        q2 = input("Q2. 내원 당시 환자 상태/검사(증상 중심): ").strip()
        q3_imgs = self._input_image_pairs("Q3. 내원 시 촬영 이미지", save_name=save_name)
        q4 = input("Q4. 치료 내용(과정/재료/횟수 등 진료 중심): ").strip()
        q5_imgs = self._input_image_pairs("Q5. 치료 중/후 이미지", save_name=save_name)
        q6 = input("Q6. 치료 결과/예후/주의사항: ").strip()
        q7_imgs = self._input_image_pairs("Q7. 결과 이미지", save_name=save_name)
        q8 = input("Q8. 기타 강조사항(통증/심미/기능 등): ").strip()
        return {
            "question1_concept": q1,
            "question2_condition": q2,
            "question3_visit_images": q3_imgs,
            "question4_treatment": q4,
            "question5_therapy_images": q5_imgs,
            "question6_result": q6,
            "question7_result_images": q7_imgs,
            "question8_extra": q8,
        }

    # ---------- collect (핵심 엔트리) ----------
    def collect(self, mode: str = "use") -> dict:
        # 외부 주입이 있다면 스키마 보정 + 종료
        if self.input_data:
            save_name = (self.input_data.get("hospital") or {}).get("save_name", "")
            self.input_data.setdefault("question3_visit_images", self.input_data.pop("visit_images", []))
            self.input_data.setdefault("question5_therapy_images", self.input_data.pop("therapy_images", []))
            self.input_data.setdefault("question7_result_images", self.input_data.pop("result_images", []))
            self.input_data.setdefault("case_id", _gen_case_id(save_name))
            self.input_data["clinical_context"] = self._build_clinical_context(self.input_data)
            return self._finalize_and_save(self.input_data, mode=mode)

        # 1) 병원 정보
        use_manual = input("병원 정보를 수동 입력하시겠습니까? (Y/N): ").strip().lower() == "y"
        if use_manual:
            hospital_info = self.manual_input_hospital_info()
        else:
            hospital_name = input("병원 이름을 입력하세요: ").strip()
            hospital_info = self.interactive_select_hospital(hospital_name)

        region_info = self.extract_region_info(hospital_info.get("address", ""))
        save_name = hospital_info.get("save_name", "") or "hospital"

        # 2) 로고/명함 자동 탐색 보강
        if not hospital_info.get("logo"):
            ask_logo = input("로고 파일명이 없습니다. 추가하시겠습니까? (Y/N): ").strip().lower() == "y"
            if ask_logo:
                logo_file = input("로고 이미지 파일명: ").strip()
                self.process_uploaded_images({logo_file: f"{save_name}_logo"})
                hospital_info["logo"] = self.find_image_file(save_name, "_logo")
        if not hospital_info.get("business_card"):
            ask_card = input("명함 파일명이 없습니다. 추가하시겠습니까? (Y/N): ").strip().lower() == "y"
            if ask_card:
                card_file = input("명함 이미지 파일명: ").strip()
                self.process_uploaded_images({card_file: f"{save_name}_business_card"})
                hospital_info["business_card"] = self.find_image_file(save_name, "_business_card")

        # 3) 모드 분기
        if mode == "test":
            # 자동(테스트케이스) / 수동 중 선택 흐름은 "입력 방식"에서 이미 결정됨
            if not self.test_data_path.exists():
                print(f"⚠️ 테스트 입력 파일이 없어 수동 흐름으로 전환합니다: {self.test_data_path}")
                return self._collect_use_like_flow(hospital_info, region_info, save_name, mode)

            # 자동 입력: 케이스 번호를 입력받고 해당 케이스에서 Q값/카테고리를 가져옴
            auto_or_manual = input("테스트케이스를 사용할까요? (Y=케이스에서 로드 / N=수동 입력): ").strip().lower()
            if auto_or_manual == "y":
                with open(self.test_data_path, encoding="utf-8") as f:
                    data_all = json.load(f)
                self.case_num = input("테스트 케이스 번호 입력 (기본: 1): ").strip() or "1"
                while True:
                    case_key = f"test_case_{self.case_num}"
                    if case_key in data_all:
                        break
                    self.case_num = input("해당 케이스 번호가 존재하지 않습니다. 다시 입력: ").strip() or "1"
                legacy = dict(data_all[case_key])
                case_new = migrate_legacy_case_to_new_schema(legacy)

                # 카테고리: 케이스에 있으면 사용, 없으면 수동 입력
                category = case_new.get("category", "").strip()
                if not category:
                    print("\n⚠️ 케이스에 카테고리가 없습니다. 직접 입력합니다.")
                    category = self._input_category()

                # S/P/T
                spt = self.select_spt(category)

                # 치식 여부/입력
                include_teeth_any = input("치식 번호를 포함하시겠습니까? (Y/N): ").strip().lower() == "y"
                tooth_numbers: List[str] = []
                if include_teeth_any:
                    tooth_numbers = self._input_tooth_numbers("FDI 2자리를 콤마로 입력하세요 (예: 11, 21): ")

                # 페르소나
                rep_personas = self.get_representative_personas(category) if category else []
                selected_personas = self.select_personas(rep_personas) if rep_personas else []

                data = {
                    "hospital": hospital_info,
                    **region_info,
                    "category": category,
                    **spt,
                    "include_tooth_numbers": include_teeth_any,
                    "tooth_numbers": tooth_numbers,
                    # Q1~Q8 (케이스에서 수급)
                    "question1_concept": case_new.get("question1_concept", ""),
                    "question2_condition": case_new.get("question2_condition", ""),
                    "question3_visit_images": case_new.get("question3_visit_images", []),
                    "question4_treatment": case_new.get("question4_treatment", ""),
                    "question5_therapy_images": case_new.get("question5_therapy_images", []),
                    "question6_result": case_new.get("question6_result", ""),
                    "question7_result_images": case_new.get("question7_result_images", []),
                    "question8_extra": case_new.get("question8_extra", ""),
                    # 페르소나
                    "persona_candidates": selected_personas or rep_personas,
                    "representative_persona": (selected_personas[0] if selected_personas else (rep_personas[0] if rep_personas else "")),
                }
                data["clinical_context"] = self._build_clinical_context(data)
                return self._finalize_and_save(data, mode="test")

            # 수동 입력 흐름 (test에서도 허용)
            return self._collect_use_like_flow(hospital_info, region_info, save_name, mode)

        # use 모드: 항상 카테고리/질문 수동
        return self._collect_use_like_flow(hospital_info, region_info, save_name, mode)

    # ---------- 카테고리 수동 입력 ----------
    def _input_category(self) -> str:
        while True:
            if self.valid_categories:
                print("\n📚 사용 가능한 카테고리:", ", ".join(self.valid_categories))
            category = input("카테고리를 입력하세요(비워도 됨, 엔터): ").strip()
            if not self.valid_categories or not category or category in self.valid_categories:
                return category
            print(f"잘못된 카테고리입니다. 선택 가능: {self.valid_categories}")

    # ---------- use와 동일한 수동 플로우(test에서도 사용 가능) ----------
    def _collect_use_like_flow(self, hospital_info: dict, region_info: dict, save_name: str, mode: str) -> dict:
        # 카테고리
        category = self._input_category()

        # S/P/T
        spt = self.select_spt(category) if category else {"selected_symptom": "", "selected_procedure": "", "selected_treatment": ""}

        # 치식
        include_teeth_any = input("치식 번호를 포함하시겠습니까? (Y/N): ").strip().lower() == "y"
        tooth_numbers: List[str] = []
        if include_teeth_any:
            tooth_numbers = self._input_tooth_numbers("FDI 2자리를 콤마로 입력하세요 (예: 11, 21): ")

        # 질문 순서 Q1→Q2→Q3→Q4→Q5→Q6→Q7→Q8
        qpack = self._manual_questions_q1_to_q8(save_name=save_name)

        # 페르소나
        rep_personas = self.get_representative_personas(category) if category else []
        selected_personas = self.select_personas(rep_personas) if rep_personas else []

        data = {
            "hospital": hospital_info,
            **region_info,
            "category": category,
            **spt,
            "include_tooth_numbers": include_teeth_any,
            "tooth_numbers": tooth_numbers,
            **qpack,
            "persona_candidates": selected_personas or rep_personas,
            "representative_persona": (selected_personas[0] if selected_personas else (rep_personas[0] if rep_personas else "")),
        }
        data["clinical_context"] = self._build_clinical_context(data)
        return self._finalize_and_save(data, mode=mode)

# ------------------------------
# 엔트리포인트
# ------------------------------
if __name__ == "__main__":
    print("\n🔍 InputAgent 시작")
    print("test/use 공통 파이프라인: 병원 → 로고/명함 → 카테고리 → 증상/진료/치료 → 치식 → Q1~Q8 → 컨텍스트 → 저장/로그")
    mode = input("모드를 선택하세요 ('test' 또는 'use', 기본값 'use'): ").strip().lower() or "use"
    if mode not in ("test", "use"):
        print("잘못된 모드입니다. 기본값 'use'로 진행합니다.")
        mode = "use"
    agent = InputAgent(case_num="1")
    result = agent.collect(mode=mode)
    print("\n" + "=" * 80)
    print("📋 [INPUT RESULT]")
    print("=" * 80)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 80)
