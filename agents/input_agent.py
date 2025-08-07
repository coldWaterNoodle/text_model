import json
import shutil
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import pandas as pd

class InputAgent:
    def __init__(
        self,
        input_data: Optional[dict] = None,
        case_num: str = "1",
        test_data_path: str = "test_data/test_input_onlook.json",
        persona_csv_path: str = "test_data/persona_table.csv",
        hospital_info_path: str = "test_data/test_hospital_info.json",
        hospital_image_path: str = "test_data/hospital_image"
    ):
        self.case_num = case_num
        self.test_data_path = Path(test_data_path)
        self.input_data = input_data
        self.persona_df = pd.read_csv(persona_csv_path).fillna("")
        self.hospital_info_path = Path(hospital_info_path)
        self.hospital_image_path = Path(hospital_image_path)
        self.valid_categories = self.persona_df["카테고리"].unique().tolist()

    def collect(self, mode: str = "use") -> dict:
        if self.input_data:
            self.input_data["persona_candidates"] = self.get_persona(self.input_data.get("category", ""))
            return self.input_data

        use_manual = input("병원 정보를 수동 입력하시겠습니까? (Y/N): ").strip().lower() == "y"
        if use_manual:
            hospital_info = self.manual_input_hospital_info()
        else:
            hospital_name = input("병원 이름을 입력하세요: ").strip()
            hospital_info = self.load_hospital_info(hospital_name)
            if not hospital_info:
                print(f"'{hospital_name}'에 대한 병원 정보가 없어 직접 입력으로 진행합니다.")
                hospital_info = self.manual_input_hospital_info(hospital_name)

        if mode == "test":
            if not self.test_data_path.exists():
                raise FileNotFoundError(f"테스트 입력 파일을 찾을 수 없습니다: {self.test_data_path}")

            with open(self.test_data_path, encoding="utf-8") as f:
                data = json.load(f)

            self.case_num = input("테스트 케이스 번호 입력 (기본: 1): ").strip() or "1"

            while True:
                case_key = f"test_case_{self.case_num}"
                if self.case_num == "0":
                    print("0이 입력되어 기본 입력으로 넘어갑니다.")
                    return self.manual_input(hospital_info)
                if case_key in data:
                    break
                self.case_num = input("해당 케이스 번호가 존재하지 않거나 0입니다. 다시 입력하세요: ").strip() or "1"

            input_data = data[case_key]
            input_data["hospital"] = hospital_info

            category = input_data.get("category", "")
            if category not in self.valid_categories:
                raise ValueError(f"'{category}'는 유효한 카테고리가 아닙니다. 다음 중 하나를 선택하세요: {self.valid_categories}")

            all_personas = self.get_persona(category)
            print(f"선택 가능한 페르소나: {all_personas}")
            while True:
                selected = input("사용할 페르소나를 쉼표로 입력하거나, 엔터를 눌러 기본값 사용: ").strip()
                if not selected:
                    selected_personas = all_personas
                    break
                selected_personas = [p.strip().lower() for p in selected.split(",") if p.strip()]

                valid_aliases = {
                    p.lower(): [
                        p.lower(),
                        p.split("(")[0].strip().lower(),
                        p[p.find("(")+1:p.find(")")].strip().lower() if "(" in p and ")" in p else ""
                    ] for p in all_personas
                }

                flat_valid = set(alias for aliases in valid_aliases.values() for alias in aliases if alias)
                invalid = [p for p in selected_personas if p not in flat_valid]
                if not invalid:
                    selected_personas = [p for key in selected_personas for p, aliases in valid_aliases.items() if key in aliases]
                    break
                print(f"잘못된 페르소나가 포함되어 있습니다: {invalid}. 다시 선택해주세요.")

            input_data["persona_candidates"] = selected_personas

            ordered_data = {
                "hospital": input_data.pop("hospital"),
                **{k: input_data[k] for k in [
                    "category", "question1_concept", "question2_condition",
                    "question3_visit_photo", "question4_treatment", "question5_therapy_photo",
                    "question6_result", "question7_result_photo", "question8_extra"
                ]},
                "persona_candidates": input_data["persona_candidates"]
            }

            region_info = self.extract_region_info(hospital_info.get("address", ""))
            ordered_data.update(region_info)

            return ordered_data

        return self.manual_input(hospital_info)

    def manual_input_hospital_info(self, name: Optional[str] = None) -> dict:
        print("\n[병원 정보 수동 입력 시작]")
        name = name or input("병원 이름을 입력하세요: ").strip()
        save_name = input("병원 저장명을 입력하세요 (예: hani): ").strip()
        homepage = input("홈페이지 URL을 입력하세요: ").strip()
        phone = input("전화번호를 입력하세요: ").strip()
        address = input("병원 주소를 입력하세요 (예: 서울특별시 강남구 논현동 123): ").strip()
        map_link = input("네이버 지도 URL을 입력하세요 (없으면 Enter): ").strip() or None

        # 이미지 매핑 수동 입력 받기
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

        return {
            "name": name,
            "save_name": save_name,
            "homepage": homepage,
            "phone": phone,
            "address": address,
            "map_link": map_link,
            "logo": logo,
            "business_card": business_card
        }

    def process_uploaded_images(
        self,
        mapping: dict,
        test_image_dir: Path = Path("test_data/test_image"),
        hospital_image_dir: Path = Path("test_data/hospital_image")
    ) -> None:
        hospital_image_dir.mkdir(parents=True, exist_ok=True)
        for original_filename, mapped_stem in mapping.items():
            original_path = test_image_dir / original_filename
            if not original_path.exists():
                print(f"❌ 파일 없음: {original_filename}")
                continue
            ext = original_path.suffix.lower()
            new_filename = f"{mapped_stem}{ext}"
            new_path = hospital_image_dir / new_filename
            try:
                shutil.copy(original_path, new_path)
                print(f"✅ {original_filename} → {new_filename} 복사 완료")
            except Exception as e:
                print(f"⚠️ 복사 실패: {original_filename} → {new_filename} | {e}")

    def get_persona(self, category: str) -> list:
        row = self.persona_df[self.persona_df["카테고리"] == category]
        if not row.empty:
            personas = row.iloc[0]["대표페르소나"]
            return [p.strip() for p in personas.split(",") if p.strip()]
        return []

    def load_hospital_info(self, name: str) -> Optional[dict]:
        if not self.hospital_info_path.exists():
            return None
        with open(self.hospital_info_path, encoding="utf-8") as f:
            hospital_list = json.load(f)
        for h in hospital_list:
            if h["name"] == name:
                save_name = h.get("save_name", name)
                h["logo"] = self.find_image_file(save_name, "_logo")
                h["business_card"] = self.find_image_file(save_name, "_business_card")
                return h
        return None

    def find_image_file(self, name: str, keyword: str) -> Optional[str]:
        for ext in ["png", "jpg", "jpeg", "webp"]:
            for file in self.hospital_image_path.glob(f"{name}{keyword}.{ext}"):
                return file.name
        return None

    def extract_region_info(self, address: str) -> dict:
        parts = address.split()
        if len(parts) < 2:
            return {"city": "", "district": "", "region_phrase": ""}
        city = parts[0].replace("특별시", "").replace("광역시", "").replace("자치시", "").replace("도", "")
        district = parts[1].replace("시", "").replace("군", "").replace("구", "")
        return {
            "city": city,
            "district": district,
            "region_phrase": f"{city} {district}"
        }

    def manual_input(self, hospital_info: dict) -> dict:
        while True:
            category = input("카테고리를 입력하세요: ").strip()
            if category in self.valid_categories:
                break
            print(f"잘못된 카테고리입니다. 다음 중 하나를 선택하세요: {self.valid_categories}")

        all_personas = self.get_persona(category)
        selected_personas = []
        if all_personas:
            print(f"선택 가능한 페르소나: {all_personas}")
            while True:
                selected = input("사용할 페르소나를 쉼표로 입력하거나, 엔터를 눌러 건너뛰세요: ").strip()
                if not selected:
                    break
                selected_personas = [p.strip().lower() for p in selected.split(",") if p.strip()]
                valid_aliases = {
                    p.lower(): [
                        p.lower(),
                        p.split("(")[0].strip().lower(),
                        p[p.find("(")+1:p.find(")")].strip().lower() if "(" in p and ")" in p else ""
                    ] for p in all_personas
                }
                flat_valid = set(alias for aliases in valid_aliases.values() for alias in aliases if alias)
                invalid = [p for p in selected_personas if p not in flat_valid]
                if not invalid:
                    selected_personas = [p for key in selected_personas for p, aliases in valid_aliases.items() if key in aliases]
                    break
                print(f"잘못된 페르소나가 포함되어 있습니다: {invalid}. 다시 선택해주세요.")

        region_info = self.extract_region_info(hospital_info.get("address", ""))

        return {
            "hospital": hospital_info,
            "category": category,
            "persona_candidates": selected_personas or all_personas,
            "question1_concept":     input("Q1. 질환 개념 및 강조 메시지: ").strip(),
            "question2_condition":   input("Q2. 내원 당시 환자 상태: ").strip(),
            "question3_visit_photo": self.input_images_with_description("Q3"),
            "question4_treatment":   input("Q4. 치료 내용: ").strip(),
            "question5_therapy_photo": self.input_images_with_description("Q5"),
            "question6_result":      input("Q6. 치료 결과 메시지: ").strip(),
            "question7_result_photo": self.input_images_with_description("Q7"),
            "question8_extra":       input("Q8. 기타 강조사항: ").strip(),
            **region_info
        }

    def input_images_with_description(self, question_num: str) -> list:
        images = []
        print(f"{question_num}번 질문에 해당하는 이미지를 추가합니다.")
        while True:
            filename = input("파일명을 입력하세요: ").strip()
            if not filename:
                break
            desc = input("해당 이미지 설명을 입력하세요: ").strip()
            images.append({"file": filename, "desc": desc})
            more = input("이미지를 더 추가하시겠습니까? (y/n): ").strip().lower()
            if more != "y":
                break
        return images

    def save_log(self, result: dict, mode: str = "use") -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(f"test_logs/{mode}")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{now}_input_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mode = input("모드를 선택하세요 ('test' 또는 'use', 기본값 'use'): ").strip().lower() or "use"
    if mode not in ("test", "use"):
        print("잘못된 모드입니다. 기본값 'use'로 진행합니다.")
        mode = "use"

    case_num = "1"
    agent = InputAgent(case_num=case_num)
    result = agent.collect(mode=mode)
    agent.save_log(result, mode=mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
