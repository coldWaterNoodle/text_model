# input_agent.py
# -*- coding: utf-8 -*-

"""
InputAgent (병원 입력/이미지 업로드 보강 + 정리판)
- 기능: 병원 선택/입력, 카테고리·증상/진료/치료 유도 선택, Q1~Q8(테스트 로그/직접 입력),
  이미지 탐색·정규화 복사, 지역(city/district) 추출, 페르소나 선택, 필수사실(치식/날짜/횟수/장비) 추출,
  결과 검증 및 로그 저장
- 보강:
  * 수동 병원 입력 시 로고/명함 파일명 입력란 제공
  * 원본 파일을 여러 폴더(절대경로/현재경로/images/test_image/hospital_image)에서 대소문자 무시로 탐색
  * 찾은 원본을 규칙명 {save_name}_{원파일명(확장자제외)}_{logo|business_card}.{ext} 으로 hospital_image에 복사
  * 자동 매핑 실패 시에도 즉시 업로드 입력을 요청
  * run_test(): 병원 정보 수동 입력 폴백 허용
"""

from __future__ import annotations

import json
import os
import re
import sys
import difflib
import shutil
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


# ======================================================================
# 공용 유틸
# ======================================================================

ENCODINGS = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]


def read_csv_kr(path: Union[str, Path]) -> pd.DataFrame:
    """KR 인코딩 강인 CSV 로더"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")
    last_err = None
    for enc in ENCODINGS:
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


def ensure_url(u: str) -> str:
    u = (u or "").strip()
    if u and not re.match(r"^https?://", u, flags=re.I):
        u = "https://" + u
    return u


def dedup_keep_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for x in items or []:
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ======================================================================
# CategoryDataIndex: category_data.csv → 카테고리/증상/진료/치료 트리
# ======================================================================

class CategoryDataIndex:
    TOKEN_RE = re.compile(r"[^가-힣A-Za-z0-9\s]")

    def __init__(self, category_csv_path: str = "test_data/category_data.csv"):
        self.category_csv_path = Path(category_csv_path)
        self.tree: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        self._build_tree()

    # ---------- Public API ----------
    def categories(self) -> List[str]:
        return sorted(list(self.tree.keys()))

    def symptoms_for(self, category: str) -> List[Tuple[str, str]]:
        cat = self.tree.get(category, {})
        return [(self._summarize_label(sym), sym) for sym in sorted(cat.keys())]

    def procedures_for(self, category: str, symptom_text: str) -> List[Tuple[str, str]]:
        procs = self.tree.get(category, {}).get(symptom_text, {})
        return [(self._summarize_label(p), p) for p in sorted(procs.keys())]

    def treatments_for(self, category: str, symptom_text: str, procedure_text: str) -> List[Tuple[str, str]]:
        txs = self.tree.get(category, {}).get(symptom_text, {}).get(procedure_text, [])
        out, seen = [], set()
        for t in txs:
            if t not in seen:
                seen.add(t)
                out.append((self._summarize_label(t), t))
        return out

    # ---------- Internal ----------
    def _build_tree(self) -> None:
        if not self.category_csv_path.exists():
            self.tree = {}
            return
        df = read_csv_kr(self.category_csv_path)
        for col in ["카테고리", "증상", "진료", "치료"]:
            if col not in df.columns:
                df[col] = ""
        df = df.assign(
            카테고리=df["카테고리"].map(lambda x: str(x).strip()),
            증상=df["증상"].map(lambda x: str(x).strip()),
            진료=df["진료"].map(lambda x: str(x).strip()),
            치료=df["치료"].map(lambda x: str(x).strip()),
        )

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
        out, seen = [], set()
        for w in toks:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= max_tokens:
                break
        return " / ".join(out) if out else text[:20]


# ======================================================================
# 데이터 모델
# ======================================================================

@dataclass
class Hospital:
    name: str = ""
    save_name: str = ""
    phone: str = ""
    address: str = ""
    homepage: str = ""
    map_link: str = ""
    logo: Optional[str] = None
    business_card: Optional[str] = None


@dataclass
class ImagePair:
    filename: str
    description: str


# ======================================================================
# InputAgent
# ======================================================================

class InputAgent:
    # 병원명 정규화 시 제거 접미사
    SUFFIXES = ["치과의원", "치과병원", "치과", "의원", "병원", "의료원", "메디컬센터", "센터", "클리닉", "덴탈"]

    # 질문 이미지 소스 탐색 기본 루트
    IMAGE_SEARCH_DIRS = [
        ".",                       # 현재 경로
        "images",                  # images/
        "test_data/test_image",    # 과거/테스트 소스 위치
        "test_data/hospital_image" # 혹시 섞인 경우
    ]

    def __init__(
        self,
        test_data_path: str = "test_data/test_input_onlook.json",
        persona_csv_path: str = "test_data/persona_table.csv",
        category_csv_path: str = "test_data/category_data.csv",
        hospital_info_path: str = "test_data/test_hospital_info.json",
    ):
        # 경로/리소스
        self.test_data_path = Path(test_data_path)
        self.hospital_info_path = Path(hospital_info_path)
        self.persona_df = read_csv_kr(persona_csv_path) if Path(persona_csv_path).exists() else pd.DataFrame()
        self.category_index = CategoryDataIndex(category_csv_path)
        self.valid_categories = (
            sorted(self.persona_df["카테고리"].unique().tolist())
            if (not self.persona_df.empty and "카테고리" in self.persona_df.columns)
            else []
        )

    # ------------------------------------------------------------------
    # 공용 정규화/추출 유틸
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(s: str) -> str:
        if not isinstance(s, str):
            return ""
        return re.sub(r"\s+", " ", s.strip())

    def _normalize_hospital_name(self, s: str) -> str:
        """병원명 → save_name 후보"""
        s = re.sub(r"\(.*?\)", "", (s or "").strip().lower())
        s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
        changed = True
        while changed and s:
            changed = False
            for suf in self.SUFFIXES:
                if s.endswith(suf):
                    s = s[: -len(suf)]
                    changed = True
        return s

    def _derive_region(self, address: str) -> Tuple[str, str, str]:
        """
        address에서 city, district, region_phrase 추출(라이트 룰)
        예) '서울특별시 강남구 ...' -> ('서울', '강남구', '서울 강남구권')
            '경기도 화성시 ...' -> ('경기', '화성시', '경기 화성시권')
        """
        addr = address or ""
        # 광역/특별시
        m = re.search(r"(서울특별시|부산광역시|대구광역시|인천광역시|광주광역시|대전광역시|울산광역시)", addr)
        if m:
            city_map = {
                "서울특별시": "서울", "부산광역시": "부산", "대구광역시": "대구", "인천광역시": "인천",
                "광주광역시": "광주", "대전광역시": "대전", "울산광역시": "울산"
            }
            city = city_map.get(m.group(1), "")
        else:
            # 도
            m2 = re.search(r"(경기도|강원도|충청북도|충청남도|전라북도|전라남도|경상북도|경상남도|제주특별자치도)", addr)
            dmap = {
                "경기도": "경기", "강원도": "강원", "충청북도": "충북", "충청남도": "충남",
                "전라북도": "전북", "전라남도": "전남", "경상북도": "경북", "경상남도": "경남", "제주특별자치도": "제주"
            }
            city = dmap.get(m2.group(1), "") if m2 else ""
        m3 = re.search(r"([가-힣]+구|[가-힣]+시|[가-힣]+군)", addr)
        district = m3.group(1) if m3 else ""
        region_phrase = f"{city} {district}권".strip().replace("  ", " ")
        return city, district, region_phrase

    @staticmethod
    def _extract_must_include_facts(text_blobs: List[str]) -> Dict[str, List[str]]:
        """
        - 치식(FDI 2자리): 11~18, 21~28, 31~38, 41~48
        - 날짜: YYYY.MM(.DD) 허용
        - 횟수: 'N회'
        - 장비/키워드: 지정 리스트 매칭
        """
        all_txt = " ".join([t for t in text_blobs if isinstance(t, str)])
        fdi = sorted(set(re.findall(r"\b(?:1[1-8]|2[1-8]|3[1-8]|4[1-8])\b", all_txt)))
        dates = sorted(set(re.findall(r"\b(20\d{2}\.(?:0?[1-9]|1[0-2])(?:\.(?:0?[1-9]|[12]\d|3[01]))?)\b", all_txt)))
        counts = sorted(set(re.findall(r"\b(\d{1,2})회\b", all_txt)))
        equip_kw = []
        for kw in ["러버댐", "클램프", "CT", "파노라마", "근관확대", "Apex", "세척", "소독", "크라운", "임플란트"]:
            if kw in all_txt:
                equip_kw.append(kw)
        return {"tooth_fdi": fdi, "dates": dates, "counts": counts, "equip": equip_kw}

    # ------------------------------------------------------------------
    # 이미지 탐색/정규화 복사
    # ------------------------------------------------------------------
    def _find_source_image(self, filename: str) -> Optional[Path]:
        """
        filename이 절대/상대 경로면 그대로 확인,
        아니면 IMAGE_SEARCH_DIRS 내에서 재귀 탐색(images/날짜/ 포함).
        여러개면 mtime 최신 1개. (대/소문자 무시)
        """
        candidate = Path(filename).expanduser()
        if candidate.exists():
            return candidate

        name_only = Path(filename).name.lower()
        hits: List[Path] = []
        for root in self.IMAGE_SEARCH_DIRS:
            root_path = Path(root)
            if not root_path.exists():
                continue
            for p in root_path.rglob("*"):
                if p.is_file() and p.name.lower() == name_only:
                    hits.append(p)
        if not hits:
            return None
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0]

    # 여러 폴더에서 원본 찾기(로고/명함 업로드 전용)
    def _find_original_image_path(self, filename: str) -> Optional[Path]:
        """
        사용자가 입력한 원본 파일명을 여러 경로에서 대소문자 무시로 탐색.
        우선순위: 절대/상대경로 → 현재폴더 → images → test_data/test_image → test_data/hospital_image
        """
        if not filename:
            return None

        cand = Path(filename).expanduser()
        if cand.exists() and cand.is_file():
            return cand

        targets = [
            Path("."),
            Path("images"),
            Path("test_data/test_image"),
            Path("test_data/hospital_image"),
        ]
        name_lower = Path(filename).name.lower()

        hits: List[Path] = []
        for root in targets:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if p.is_file() and p.name.lower() == name_lower:
                    hits.append(p)

        if not hits:
            return None
        hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0]

    def _normalize_and_copy_image(
        self,
        filename: str,
        save_name: str,
        dest_dir: Union[str, Path] = "test_data/test_image",
        suffix: str = ""
    ) -> str:
        """
        - filename 실존 파일 탐색(_find_source_image)
        - 찾으면 dest_dir로 복사
        - 저장 파일명: {save_name}_{base}{suffix}{ext}
        - 반환: 저장된 '파일명'(basename) — 소스 없어도 규칙명 반환
        """
        src = self._find_source_image(filename)
        base, ext = os.path.splitext(Path(filename).name)
        target_name = f"{save_name}_{base}{suffix}{ext}"

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dst = dest_dir / target_name

        if src is None:
            print(f"⚠️ 소스 이미지가 존재하지 않습니다: {filename}")
            return target_name

        try:
            shutil.copy2(src, dst)
            print(f"✅ 이미지 복사: {src} → {dst}")
        except Exception as e:
            print(f"❌ 이미지 복사 실패: {e}")
        return target_name

    # ---------------- 이미지 업로드(로고/명함) 전용: 여러 폴더 → hospital_image ----------------
    def _process_uploaded_hospital_images(
        self,
        mapping: dict,
        dst_dir: Path = Path("test_data/hospital_image"),
    ) -> None:
        """
        mapping 예: {"바나나.jpg": f"{save_name}_logo", "명함.png": f"{save_name}_business_card"}
        저장 규칙 파일명: {save_name}_{원파일명(확장자제외)}_{logo|business_card}.{ext}
        - 원본은 절대경로/현재경로/images/test_image/hospital_image 어디에 있어도 OK
        - 대소문자 무시 매칭
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        for original_filename, mapped_stem in (mapping or {}).items():
            # 1) 원본 파일 경로 탐색 (여러 폴더에서)
            original_path = self._find_original_image_path(original_filename)
            if not original_path:
                print(f"❌ 파일을 찾을 수 없습니다: {original_filename}")
                continue

            # 2) 규칙명 산출
            base_stem = original_path.stem
            safe_base = re.sub(r"[^가-힣A-Za-z0-9_-]+", "_", base_stem).strip("_")
            suffix = "_logo" if mapped_stem.endswith("_logo") else "_business_card"
            save_name = mapped_stem.split("_")[0]
            ext = original_path.suffix.lower()
            new_filename = f"{save_name}_{safe_base}{suffix}{ext}"
            new_path = dst_dir / new_filename

            # 3) 복사(동일 폴더여도 안전하게 복사)
            try:
                shutil.copy2(original_path, new_path)
                print(f"✅ {original_path.name} → {new_filename} 복사 완료")
            except Exception as e:
                print(f"⚠️ 복사 실패: {original_path.name} → {new_filename} | {e}")

    # ------------------------------------------------------------------
    # 병원 로딩/선택/수동입력 + 이미지 매핑
    # ------------------------------------------------------------------
    def _load_hospitals_list(self) -> List[Dict[str, str]]:
        if not self.hospital_info_path.exists():
            return []
        try:
            with open(self.hospital_info_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _attach_hospital_images(self, hospital: Dict[str, str], allow_cli: bool = False) -> Dict[str, str]:
        """
        test_data/hospital_image 자동 매핑 → 없으면 (allow_cli=True) 수동 입력.
        이미 hospital에 logo/business_card가 있으면 그대로 둠.
        """
        image_dir = Path("test_data/hospital_image")
        save_name = hospital.get("save_name") or self._normalize_hospital_name(hospital.get("name", "")) or "hospital"
        updated = dict(hospital)

        if updated.get("logo") and updated.get("business_card"):
            return updated

        auto_logo = updated.get("logo")
        auto_bc = updated.get("business_card")

        if image_dir.exists():
            if not auto_logo:
                logos = list(image_dir.glob(f"{save_name}_*_logo.*"))
                if logos:
                    auto_logo = logos[0].name
                    print(f"✅ 로고 자동 매핑: {auto_logo}")
            if not auto_bc:
                bcs = list(image_dir.glob(f"{save_name}_*_business_card.*"))
                if bcs:
                    auto_bc = bcs[0].name
                    print(f"✅ 명함 자동 매핑: {auto_bc}")

        updated["logo"] = auto_logo
        updated["business_card"] = auto_bc

        if allow_cli and (not updated["logo"] or not updated["business_card"]):
            print("\n🖼️ 병원 로고/명함 이미지를 입력해 주세요. (test_data/test_image 또는 hospital_image에 파일이 있어야 합니다)")
            logo_file, bc_file = self._input_hospital_images(save_name)
            if logo_file and not updated["logo"]:
                updated["logo"] = logo_file
            if bc_file and not updated["business_card"]:
                updated["business_card"] = bc_file

        return updated

    # ---------- 로고/명함 입력란 + 업로드 처리 ----------
    def _input_hospital_images(self, save_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        로고/명함 파일명을 입력받아,
        - hospital_image 폴더에 규칙명 파일이 있는지 확인
        - 없으면 여러 폴더에서 원본을 찾아 규칙명으로 복사
        반환: 규칙명 파일명(존재 시)
        """
        dst_dir = Path("test_data/hospital_image")

        def _ask(tag: str, suffix_word: str) -> Optional[str]:
            raw = input(f"{tag} 이미지 파일명을 입력해 주세요 (예: logo.png, 없으면 엔터): ").strip()
            if not raw:
                return None

            # 규칙명 후보
            base, ext = os.path.splitext(Path(raw).name)
            safe_base = re.sub(r"[^가-힣A-Za-z0-9_-]+", "_", base).strip("_")
            normalized = f"{save_name}_{safe_base}{suffix_word}{ext}"
            dst_path = dst_dir / normalized

            # 1) 규칙명 파일이 이미 있는지 확인
            if dst_path.exists():
                print(f"✅ {tag} 이미지 확인: {dst_path}")
                return normalized

            # 2) 없으면 원본 찾아 복사 (여러 폴더 탐색)
            mapping = {raw: f"{save_name}{suffix_word}"}
            self._process_uploaded_hospital_images(mapping, dst_dir=dst_dir)
            if dst_path.exists():
                return normalized

            print(f"⚠️ {tag} 이미지를 찾거나 복사하지 못했습니다: {raw}")
            return None

        logo_file = _ask("로고", "_logo")
        bc_file   = _ask("명함", "_business_card")
        return logo_file, bc_file

    def _input_hospital_manual(self, prefill_name: str = "", hospitals: List[Dict[str, str]] | None = None) -> Dict[str, str]:
        """수동으로 병원 전체 정보 입력 (로고/명함 업로드 포함)"""
        hospitals = hospitals or []
        names = [h.get("name", "") for h in hospitals]
        norm_names = [self._normalize_hospital_name(n) for n in names]

        while True:
            print("\n[병원 정보 입력]")
            name = input("병원명 : ").strip() or prefill_name
            if not name:
                print("⚠️ 병원명은 필수입니다. 다시 입력해 주세요.")
                continue

            q = self._normalize_hospital_name(name)
            # 동일/유사 병원 안내
            if norm_names and q in norm_names:
                exist = hospitals[norm_names.index(q)]
                yn = input(f"❗ 이미 등록된 병원입니다: '{exist.get('name','')}'. 사용하시겠습니까? (Y/N): ").strip().lower()
                if yn == "y":
                    if not exist.get("save_name"):
                        exist = {**exist, "save_name": self._normalize_hospital_name(exist.get("name", ""))}
                    exist = self._attach_hospital_images(exist, allow_cli=False)
                    return exist
            else:
                close = difflib.get_close_matches(q, norm_names, n=1, cutoff=0.7) if norm_names else []
                if close:
                    cand = hospitals[norm_names.index(close[0])]
                    yn = input(f"❗ 비슷한 병원이 있습니다: '{cand.get('name','')}'. 사용하시겠습니까? (Y/N): ").strip().lower()
                    if yn == "y":
                        if not cand.get("save_name"):
                            cand = {**cand, "save_name": self._normalize_hospital_name(cand.get("name", ""))}
                        cand = self._attach_hospital_images(cand, allow_cli=False)
                        return cand

            save_name = input("저장용 병원명(save_name, 영문/숫자/소문자 권장, 미입력 시 자동 생성): ").strip() or self._normalize_hospital_name(name)
            phone = input("전화번호: ").strip()
            address = input("주소: ").strip()
            homepage = ensure_url(input("홈페이지 URL: ").strip())
            map_link = ensure_url(input("지도 URL: ").strip())

            # ➕ 로고/명함 파일명 직접 입력란
            print("\n[병원 이미지 업로드/매핑]")
            print(" - test_data/test_image 또는 test_data/hospital_image 폴더에 원본 파일을 두신 뒤 파일명을 입력해 주세요.")
            print(" - 입력하시면 규칙명으로 test_data/hospital_image 폴더에 복사됩니다.")
            input_logo = input("로고 파일명 (예: 바나나.jpg, 없으면 엔터): ").strip()
            input_card = input("명함 파일명 (예: 바나나.jpg, 없으면 엔터): ").strip()

            temp = {
                "name": name,
                "save_name": save_name,
                "phone": phone,
                "address": address,
                "homepage": homepage,
                "map_link": map_link
            }

            # 1차 자동 매핑
            temp = self._attach_hospital_images(temp, allow_cli=False)

            # 업로드 입력이 있으면 즉시 복사/리네이밍 수행
            mapping = {}
            if input_logo:
                mapping[input_logo] = f"{save_name}_logo"
            if input_card:
                mapping[input_card] = f"{save_name}_business_card"
            if mapping:
                self._process_uploaded_hospital_images(mapping)

            # 복사 후 최종 규칙명 재탐색
            image_dir = Path("test_data/hospital_image")
            if not temp.get("logo"):
                cand = list(image_dir.glob(f"{save_name}_*_logo.*"))
                if cand:
                    temp["logo"] = cand[0].name
            if not temp.get("business_card"):
                cand = list(image_dir.glob(f"{save_name}_*_business_card.*"))
                if cand:
                    temp["business_card"] = cand[0].name

            print("\n📌 입력 요약")
            for k in ["name", "save_name", "phone", "address", "homepage", "map_link", "logo", "business_card"]:
                print(f"- {k}: {temp.get(k) or '(없음)'}")
            if input("이대로 등록할까요? (Y/N): ").strip().lower() == "y":
                return temp
            print("다시 입력을 진행하겠습니다.")
            prefill_name = name

    def _select_hospital(self, allow_manual: bool = True) -> Dict[str, str]:
        hospitals = self._load_hospitals_list()
        if not hospitals:
            if allow_manual:
                return self._input_hospital_manual()
            raise FileNotFoundError("hospital_info.json에 등록된 병원이 없습니다.")

        print("\n🏥 선택 가능한 병원:")
        for i, h in enumerate(hospitals, 1):
            addr = h.get("address", "")
            print(f"{i}. {h.get('name','')} ({addr})")

        names = [h.get("name", "") for h in hospitals]
        norm_names = [self._normalize_hospital_name(n) for n in names]

        while True:
            choice = input(f"병원 번호(1-{len(hospitals)}) 또는 이름 직접 입력: ").strip()
            if not choice:
                print("⚠️ 입력이 비었습니다. 다시 시도해 주세요.")
                continue

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(hospitals):
                    chosen = hospitals[idx]
                    if not chosen.get("save_name"):
                        chosen = {**chosen, "save_name": self._normalize_hospital_name(chosen.get("name", ""))}
                    return self._attach_hospital_images(chosen, allow_cli=False)
                print(f"⚠️ 1-{len(hospitals)} 범위로 입력해 주세요.")
                continue

            q = self._normalize_hospital_name(choice)

            if q in norm_names:
                chosen = hospitals[norm_names.index(q)]
                if not chosen.get("save_name"):
                    chosen = {**chosen, "save_name": self._normalize_hospital_name(chosen.get("name", ""))}
                return self._attach_hospital_images(chosen, allow_cli=False)

            partial_hits = [i for i, n in enumerate(norm_names) if q in n]
            if len(partial_hits) == 1:
                chosen = hospitals[partial_hits[0]]
                if not chosen.get("save_name"):
                    chosen = {**chosen, "save_name": self._normalize_hospital_name(chosen.get("name", ""))}
                return self._attach_hospital_images(chosen, allow_cli=False)
            elif len(partial_hits) > 1:
                print("\n🔎 여러 병원이 검색되었습니다. 번호로 선택해 주세요:")
                for j, i_hit in enumerate(partial_hits, 1):
                    h = hospitals[i_hit]
                    print(f"{j}. {h.get('name','')} ({h.get('address','')})")
                sub = input(f"선택 (1-{len(partial_hits)}): ").strip()
                if sub.isdigit():
                    sub_idx = int(sub) - 1
                    if 0 <= sub_idx < len(partial_hits):
                        chosen = hospitals[partial_hits[sub_idx]]
                        if not chosen.get("save_name"):
                            chosen = {**chosen, "save_name": self._normalize_hospital_name(chosen.get("name", ""))}
                        return self._attach_hospital_images(chosen, allow_cli=False)
                print("⚠️ 잘못된 입력입니다.")
                continue

            close = difflib.get_close_matches(q, norm_names, n=3, cutoff=0.6)
            if close:
                print("\n🧭 유사 병원 후보:")
                cand_idx = [norm_names.index(c) for c in close]
                for j, i_hit in enumerate(cand_idx, 1):
                    h = hospitals[i_hit]
                    print(f"{j}. {h.get('name','')} ({h.get('address','')})")
                sub = input(f"선택 (1-{len(cand_idx)}) 또는 Enter로 재입력: ").strip()
                if sub.isdigit():
                    sub_idx = int(sub) - 1
                    if 0 <= sub_idx < len(cand_idx):
                        chosen = hospitals[cand_idx[sub_idx]]
                        if not chosen.get("save_name"):
                            chosen = {**chosen, "save_name": self._normalize_hospital_name(chosen.get("name", ""))}
                        return self._attach_hospital_images(chosen, allow_cli=False)
                continue

            if allow_manual and input("❗ 등록된 병원이 없습니다. 새로 입력하시겠습니까? (Y/N): ").strip().lower() == "y":
                return self._input_hospital_manual(hospitals=hospitals)
            print("목록에서 다시 선택하시거나 병원명을 다시 입력해 주세요.")

    def _get_hospital(self, allow_manual: bool) -> Dict[str, str]:
        return self._select_hospital(allow_manual=allow_manual)

    # ------------------------------------------------------------------
    # 페르소나
    # ------------------------------------------------------------------
    def get_representative_personas(self, category: str) -> List[str]:
        if not category or self.persona_df.empty or "카테고리" not in self.persona_df.columns:
            return []
        row = self.persona_df[self.persona_df["카테고리"] == category]
        if row.empty:
            return []
        rep_raw = str(row.iloc[0].get("대표페르소나", "")).strip()
        return [p.strip() for p in rep_raw.split(",") if p.strip()] if rep_raw else []

    def _pick_personas(self, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        print(f"\n선택 가능한 대표 페르소나: {candidates}")
        raw = input("사용할 페르소나를 쉼표로 입력해 주세요 (엔터=모두): ").strip()
        if not raw:
            return list(dict.fromkeys(candidates))

        def aliases_for(p: str) -> List[str]:
            base = p.split("(")[0].strip().lower()
            inner = p[p.find("(")+1:p.find(")")].strip().lower() if "(" in p and ")" in p else ""
            al = [p.lower(), base]
            if inner:
                al.append(inner)
            return dedup_keep_order(al)

        wanted = [x.strip().lower() for x in raw.split(",") if x.strip()]
        pairs = [(p, aliases_for(p)) for p in candidates]

        result = []
        for w in wanted:
            for original, aliases in pairs:
                if w in aliases and original not in result:
                    result.append(original)
        return list(dict.fromkeys(result or candidates))

    # ------------------------------------------------------------------
    # 카테고리/증상/진료/치료 유도 선택
    # ------------------------------------------------------------------
    def _pick_from_options(self, title: str, options: List[Tuple[str, str]]) -> str:
        print(f"\n📋 {title}")
        if not options:
            return input("옵션이 없습니다. 직접 입력해 주세요: ").strip()
        for i, (label, value) in enumerate(options, 1):
            short = (value[:60] + "...") if len(value) > 60 else value
            print(f"{i}. {label}  |  {short}")
        while True:
            choice = input(f"선택 (1-{len(options)}) 또는 직접입력: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx][1]
                print(f"⚠️ 1-{len(options)} 범위로 입력해 주세요.")
            elif choice:
                return choice
            else:
                print("⚠️ 입력이 비었습니다. 다시 선택해 주세요.")

    def _guided_selection(self, category: str) -> Dict[str, str]:
        sym = self._pick_from_options("[1단계] 증상 선택", self.category_index.symptoms_for(category))
        proc = self._pick_from_options("[2단계] 진료 선택", self.category_index.procedures_for(category, sym))
        tx = self._pick_from_options("[3단계] 치료 선택", self.category_index.treatments_for(category, sym, proc))
        return {"selected_symptom": sym, "selected_procedure": proc, "selected_treatment": tx}

    def _input_category(self) -> str:
        all_cats = self.valid_categories or self.category_index.categories()
        print("\n📚 사용 가능한 카테고리:")
        print(", ".join(all_cats))
        while True:
            cat = input("카테고리를 입력하세요: ").strip()
            if not cat:
                print("⚠️ 빈 값입니다. 카테고리를 입력해 주세요.")
                continue
            if self.category_index.tree and cat not in self.category_index.tree:
                print("⚠️ category_data.csv에 해당 카테고리가 없습니다. 가능한 카테고리:", ", ".join(self.category_index.categories()))
                continue
            return cat

    # ------------------------------------------------------------------
    # Q1~Q8 입력/로드
    # ------------------------------------------------------------------
    def _input_image_pairs(self, prompt_title: str, save_name: str = "") -> List[Dict[str, str]]:
        print(f"\n🖼️ {prompt_title} — 이미지 파일명과 설명을 입력해 주세요.")
        print("   (예: '바나나.jpg' 또는 'images/2025-08-11/바나나.jpg')")
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
                filename=filename, save_name=save_name, dest_dir="test_data/test_image", suffix=""
            )
            pairs.append({"filename": normalized_basename, "description": description})
        return pairs

    def _ask_questions_8(self, save_name: str) -> Dict[str, object]:
        print("\n✍️ 8개 질문을 입력합니다.")
        return {
            "question1_concept": input("Q1. 질환 개념/핵심 메시지: ").strip(),
            "question2_condition": input("Q2. 내원 당시 상태/검사(증상 중심): ").strip(),
            "visit_images": self._input_image_pairs("Q3. 내원 시 촬영 이미지", save_name=save_name),
            "question4_treatment": input("Q4. 치료 내용(과정/재료/횟수 등): ").strip(),
            "therapy_images": self._input_image_pairs("Q5. 치료 중/후 이미지", save_name=save_name),
            "question6_result": input("Q6. 치료 결과/예후/주의사항: ").strip(),
            "result_images": self._input_image_pairs("Q7. 결과 이미지", save_name=save_name),
            "question8_extra": input("Q8. 기타 강조사항(통증/심미/기능 등): ").strip(),
        }

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
                    result[k] = [
                        {
                            "filename": (img.get("filename", "") if isinstance(img, dict) else ""),
                            "description": (img.get("description", "") if isinstance(img, dict) else "")
                        } for img in imgs if isinstance(img, (dict,))
                    ]
                else:
                    result[k] = data.get(k, "")
            return result
        except Exception as e:
            print(f"⚠️ 로그 로드 실패: {e}")
            return {k: ([] if k.endswith("images") else "") for k in keys}

    # ------------------------------------------------------------------
    # 실행 플로우 (TEST / USE)
    # ------------------------------------------------------------------
    def run_test(self) -> Optional[dict]:
        # 수동 입력 폴백 허용
        hospital = self._get_hospital(allow_manual=True)
        save_name = hospital.get("save_name") or self._normalize_hospital_name(hospital.get("name", "")) or "hospital"
        hospital = self._attach_hospital_images({**hospital, "save_name": save_name}, allow_cli=True)

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

        idx = self._ask_index("\n케이스 선택", 1, len(cases))
        selected = data[cases[idx - 1]]

        category = selected.get("category", "").strip() or self._input_category()
        print(f"\n✅ 카테고리: {category}")

        picked = self._guided_selection(category)

        log_path = input("\n이전 입력 로그 파일 경로를 입력하세요 (없으면 엔터): ").strip()
        q8 = self._load_q8_from_log(log_path) if log_path else self._empty_q8()

        # 페르소나
        persona_candidates = self.get_representative_personas(category)
        selected_personas = self._pick_personas(persona_candidates)
        persona_candidates = dedup_keep_order(persona_candidates)
        selected_personas = dedup_keep_order(selected_personas)
        representative_persona = selected_personas[0] if selected_personas else (persona_candidates[0] if persona_candidates else "")

        # 파생 필드
        city, district, region_phrase = self._derive_region(hospital.get("address", ""))
        geo_branding = {"clinic_alias": hospital.get("name", ""), "region_line": f"{city} {district} 환자분들께".strip()}
        meta_panel = {"address": hospital.get("address", ""), "phone": hospital.get("phone", ""),
                      "homepage": hospital.get("homepage", ""), "map_link": hospital.get("map_link", ""), "treatment_period": ""}
        link_policy = {"homepage_in_body_once": True, "map_in_footer_only": True}
        images_index = self._index_image_refs(q8.get("visit_images", []), q8.get("therapy_images", []), q8.get("result_images", []))
        content_flow_hint = "서론 → 진단 → 치료 → 결과 → 관리(FAQ)"
        persona_structure_guide = self._persona_structure_guide(representative_persona)
        must_include_facts = self._extract_must_include_facts([
            q8.get("question1_concept",""), q8.get("question2_condition",""),
            q8.get("question4_treatment",""), q8.get("question6_result",""), q8.get("question8_extra","")
        ])

        res = {
            "mode": "test",
            "schema_version": "plan-input/1.0.0",
            "hospital": {**hospital, "save_name": save_name, "city": city, "district": district, "region_phrase": region_phrase},
            "category": category,
            **picked,
            **q8,
            "persona_candidates": persona_candidates,
            "representative_persona": representative_persona,
            "selected_personas": selected_personas,
            # Plan-ready
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

    def run_use(self) -> dict:
        hospital = self._get_hospital(allow_manual=True)
        save_name = hospital.get("save_name") or self._normalize_hospital_name(hospital.get("name", "")) or "hospital"
        hospital = self._attach_hospital_images({**hospital, "save_name": save_name}, allow_cli=True)

        category = self._input_category()
        print(f"\n✅ 카테고리: {category}")
        picked = self._guided_selection(category)

        q8 = self._ask_questions_8(save_name=save_name)

        # 페르소나
        persona_candidates = self.get_representative_personas(category)
        selected_personas = self._pick_personas(persona_candidates)
        persona_candidates = dedup_keep_order(persona_candidates)
        selected_personas = dedup_keep_order(selected_personas)
        representative_persona = selected_personas[0] if selected_personas else (persona_candidates[0] if persona_candidates else "")

        # 파생 필드
        city, district, region_phrase = self._derive_region(hospital.get("address", ""))
        geo_branding = {"clinic_alias": hospital.get("name", ""), "region_line": f"{city} {district} 환자분들께".strip()}
        # 치료기간 힌트(라이트)
        period_hint = ""
        m_period = re.search(r"(20\d{2}\.\d{1,2}\.\d{1,2})\s*[-~]\s*(20\d{2}\.\d{1,2}\.\d{1,2})",
                             q8.get("question6_result","") + " " + q8.get("question8_extra",""))
        if m_period:
            period_hint = f"{m_period.group(1)}–{m_period.group(2)}"
        meta_panel = {"address": hospital.get("address", ""), "phone": hospital.get("phone", ""),
                      "homepage": hospital.get("homepage", ""), "map_link": hospital.get("map_link", ""),
                      "treatment_period": period_hint}
        link_policy = {"homepage_in_body_once": True, "map_in_footer_only": True}
        images_index = self._index_image_refs(q8.get("visit_images", []), q8.get("therapy_images", []), q8.get("result_images", []))
        content_flow_hint = "서론 → 진단 → 치료 → 결과 → 관리(FAQ)"
        persona_structure_guide = self._persona_structure_guide(representative_persona)
        must_include_facts = self._extract_must_include_facts([
            q8.get("question1_concept",""), q8.get("question2_condition",""),
            q8.get("question4_treatment",""), q8.get("question6_result",""), q8.get("question8_extra","")
        ])

        res = {
            "mode": "use",
            "schema_version": "plan-input/1.0.0",
            "hospital": {**hospital, "save_name": save_name, "city": city, "district": district, "region_phrase": region_phrase},
            "category": category,
            **picked,
            **q8,
            "persona_candidates": persona_candidates,
            "representative_persona": representative_persona,
            "selected_personas": selected_personas,
            # Plan-ready
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

    # ------------------------------------------------------------------
    # 내부 보조
    # ------------------------------------------------------------------
    @staticmethod
    def _ask_index(title: str, start: int, end: int) -> int:
        while True:
            raw = input(f"{title} ({start}-{end}): ").strip()
            if raw.isdigit():
                v = int(raw)
                if start <= v <= end:
                    return v
            print("⚠️ 올바른 번호를 입력해 주세요.")

    @staticmethod
    def _empty_q8() -> Dict[str, object]:
        return {
            "question1_concept": "", "question2_condition": "", "visit_images": [],
            "question4_treatment": "", "therapy_images": [], "question6_result": "",
            "result_images": [], "question8_extra": ""
        }

    @staticmethod
    def _index_image_refs(visit_images: List[dict], therapy_images: List[dict], result_images: List[dict]) -> Dict[str, List[str]]:
        return {
            "visit_refs":  [f"visit_images:{i}"  for i, _ in enumerate(visit_images)],
            "therapy_refs":[f"therapy_images:{i}" for i, _ in enumerate(therapy_images)],
            "result_refs": [f"result_images:{i}" for i, _ in enumerate(result_images)],
        }

    @staticmethod
    def _persona_structure_guide(rp: str) -> str:
        if "심미" in rp: return "미적 문제→해결→변화→유지→심리효과"
        if "통증" in rp: return "통증 원인→완화→치료→예방→일상개선"
        if "기능" in rp: return "기능 문제→해결→회복→관리→일상개선"
        if "잇몸" in rp: return "건강상태→위생→치료→검진→장기관리"
        return "문제→진단→치료→결과→관리"

    def _validate_result(self, res: dict) -> None:
        assert isinstance(res.get("persona_candidates", []), list)
        assert isinstance(res.get("selected_personas", []), list)
        for k in ["visit_images", "therapy_images", "result_images"]:
            assert isinstance(res.get(k, []), list)
            for img in res.get(k, []):
                assert isinstance(img, dict) and "filename" in img and "description" in img

    # ------------------------------------------------------------------
    # CLI 저장
    # ------------------------------------------------------------------
    @staticmethod
    def save_log(res: dict, mode: str) -> Path:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("test_logs/use") if res.get("mode") == "use" else Path("test_logs/test")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{now}_input_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return log_path


# ======================================================================
# CLI
# ======================================================================

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

    path = agent.save_log(res, mode=res.get("mode", "use"))
    print(f"\n✅ 로그 저장: {path}")

    print("\n" + "=" * 80)
    print("📋 [INPUT RESULT]")
    print("=" * 80)
    print(json.dumps(res, ensure_ascii=False, indent=2, sort_keys=True), flush=True)
    print("=" * 80)
    sys.exit(0)
