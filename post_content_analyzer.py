#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post Content Analyzer

nside_with_persona.csv의 post_content를 분석하여
증상/진료/치료/카테고리로 구조화하고 저장하는 도구

AI 모델(Gemini 1.5 Flash)과 키워드 매칭을 결합한 하이브리드 방식 사용
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv


class PostContentAnalyzer:
    """
    post_content를 AI 모델로 1차 분류 후, 키워드 매칭으로 검증하는 하이브리드 분석기
    """

    def __init__(self, category_csv_path: str = "test_data/category_data.csv"):
        self.category_csv_path = Path(category_csv_path)

        # Gemini 모델 초기화
        load_dotenv()
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

        # 카테고리별 키워드 KB 구축
        self.category_kb = self._build_kb_from_csv()

    def _build_kb_from_csv(self) -> Dict[str, Dict[str, List[str]]]:
        """category_data.csv에서 카테고리별 키워드 KB 구축 (euc-kr 인코딩 사용)"""
        kb: Dict[str, Dict[str, List[str]]] = {}
        if not self.category_csv_path.exists():
            print(f"⚠️ {self.category_csv_path} 파일을 찾을 수 없습니다.")
            return kb

        try:
            # euc-kr 인코딩으로 읽기
            df = pd.read_csv(self.category_csv_path, encoding='euc-kr').fillna("")
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

            # 중복 제거
            for cat, d in kb.items():
                for f in ("symptoms", "procedures", "treatments"):
                    d[f] = self._dedup_keep_order(d[f])

            print(f"✅ {len(kb)}개 카테고리의 키워드 KB 구축 완료")
            return kb

        except Exception as e:
            print(f"❌ KB 구축 실패: {e}")
            return {}

    @staticmethod
    def _clean_text(s: str) -> str:
        """텍스트 정리"""
        if not isinstance(s, str):
            return ""
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _extract_keywords(text: str, max_tokens: int = 100) -> set:
        """텍스트에서 키워드 추출"""
        if not isinstance(text, str):
            return set()
        # 한글, 영문, 숫자만 남기고 공백으로 분리
        text = re.sub(r"[^가-힣A-Za-z0-9\s]", " ", text)
        toks = [t for t in text.split() if len(t) >= 2]
        return set(toks[:max_tokens])

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        """중복 제거하면서 순서 유지"""
        seen, out = set(), []
        for x in items or []:
            x = str(x).strip()
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def analyze_post_content(self, post_content: str) -> Dict[str, str]:
        """
        post_content를 AI 모델로 1차 분류 후, 키워드 매칭으로 검증

        Returns:
            {
                "symptoms": "추출된 증상",
                "procedures": "추출된 진료",
                "treatments": "추출된 치료",
                "category": "추출된 카테고리",
                "confidence": "신뢰도 (0.0-1.0)",
                "verification_method": "검증 방법"
            }
        """
        if not post_content or len(post_content.strip()) < 50:
            return {
                "symptoms": "", "procedures": "", "treatments": "",
                "category": "", "confidence": 0.0, "verification_method": "content_too_short"
            }

        # 1단계: AI 모델로 1차 분류
        ai_result = self._ai_classify_post_content(post_content)

        # 2단계: 키워드 매칭으로 검증
        verified_result = self._verify_with_keywords(post_content, ai_result)

        return verified_result

    def _ai_classify_post_content(self, post_content: str) -> Dict[str, str]:
        """AI 모델을 사용하여 post_content를 증상/진료/치료/카테고리로 분류"""
        try:
            prompt = f"""
다음 치과 관련 블로그 포스트 내용을 분석하여 증상, 진료, 치료, 카테고리를 추출해주세요.

포스트 내용:
{post_content[:2000]}...

다음 JSON 형식으로 응답해주세요:
{{
    "symptoms": "환자가 겪고 있는 치아 문제나 증상",
    "procedures": "의사가 진행한 진료 과정이나 계획",
    "treatments": "실제 시술이나 치료 내용",
    "category": "치료 분류 (임플란트, 심미수복, 신경치료, 충치치료, 크라운보철, 발치, 보철, 미백 중 하나)"
}}

주의사항:
- 증상은 환자가 느끼는 불편함이나 문제점
- 진료는 의사의 진단이나 치료 계획
- 치료는 실제 시술 내용
- 카테고리는 위 8개 중 가장 적합한 하나만 선택
- 각 항목은 간결하고 명확하게 작성
"""

            response = self.model.generate_content(prompt)
            result_text = response.text.strip()

            # JSON 추출 시도
            try:
                # JSON 부분만 추출
                if "{" in result_text and "}" in result_text:
                    start = result_text.find("{")
                    end = result_text.rfind("}") + 1
                    json_str = result_text[start:end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("JSON 형식이 아닙니다")

                # 필수 필드 확인
                required_fields = ["symptoms", "procedures", "treatments", "category"]
                for field in required_fields:
                    if field not in result:
                        result[field] = ""

                return result

            except (json.JSONDecodeError, ValueError) as e:
                print(f"AI 응답 파싱 실패: {e}")
                # 기본값 반환
                return {
                    "symptoms": "", "procedures": "", "treatments": "", "category": ""
                }

        except Exception as e:
            print(f"AI 분류 실패: {e}")
            return {
                "symptoms": "", "procedures": "", "treatments": "", "category": ""
            }

    def _verify_with_keywords(self, post_content: str, ai_result: Dict[str, str]) -> Dict[str, str]:
        """키워드 매칭을 통해 AI 결과를 검증하고 보정"""
        # post_content에서 키워드 추출
        content_keywords = self._extract_keywords(post_content, max_tokens=100)

        # AI 결과 검증
        verified = ai_result.copy()
        confidence = 0.0
        verification_method = "ai_only"

        # 카테고리 검증
        if ai_result.get("category"):
            category = ai_result["category"]
            if category in self.category_kb:
                # 해당 카테고리의 키워드와 매칭 확인
                cat_keywords = set()
                for field in ["symptoms", "procedures", "treatments"]:
                    cat_keywords.update(self.category_kb[category][field])

                # 키워드 매칭 점수 계산
                matches = content_keywords.intersection(cat_keywords)
                if matches:
                    confidence += 0.3
                    verification_method = "ai_keyword_verified"

                # 증상/진료/치료 필드 검증 및 보정
                for field, field_name in [("symptoms", "symptoms"), ("procedures", "procedures"), ("treatments", "treatments")]:
                    if not ai_result.get(field):
                        # AI가 추출하지 못한 경우, 키워드 매칭으로 보정
                        field_keywords = self.category_kb[category][field]
                        matched_keywords = [kw for kw in field_keywords if kw in post_content]
                        if matched_keywords:
                            verified[field] = ", ".join(matched_keywords[:3])  # 상위 3개만
                            confidence += 0.1
                            verification_method = "ai_keyword_corrected"

        # 카테고리가 없거나 잘못된 경우, 키워드 기반으로 재분류
        if not ai_result.get("category") or ai_result["category"] not in self.category_kb:
            # 키워드 기반 카테고리 스코어링
            scores = self._score_categories_by_keywords(content_keywords)
            if scores:
                best_category = max(scores, key=scores.get)
                verified["category"] = best_category
                confidence += 0.2
                verification_method = "keyword_based_fallback"

        # 최종 신뢰도 계산
        confidence = min(1.0, confidence + 0.4)  # 기본 0.4 + 검증 점수

        verified["confidence"] = round(confidence, 2)
        verified["verification_method"] = verification_method

        return verified

    def _score_categories_by_keywords(self, content_keywords: set) -> Dict[str, float]:
        """키워드 기반으로 카테고리 점수 계산"""
        scores = {}
        for category, kb_data in self.category_kb.items():
            total_matches = 0
            total_keywords = 0

            for field in ["symptoms", "procedures", "treatments"]:
                field_keywords = set(kb_data[field])
                matches = content_keywords.intersection(field_keywords)
                total_matches += len(matches)
                total_keywords += len(field_keywords)

            if total_keywords > 0:
                scores[category] = total_matches / total_keywords

        return scores

    def batch_analyze_posts(self, posts_data: List[Dict]) -> List[Dict]:
        """여러 post_content를 일괄 분석"""
        results = []
        total = len(posts_data)

        for i, post in enumerate(posts_data):
            print(f"분석 중... ({i+1}/{total})")
            post_content = post.get("post_content", "")
            if post_content:
                analysis = self.analyze_post_content(post_content)
                # 원본 데이터에 분석 결과 추가
                post.update({
                    "analyzed_symptoms": analysis["symptoms"],
                    "analyzed_procedures": analysis["procedures"],
                    "analyzed_treatments": analysis["treatments"],
                    "analyzed_category": analysis["category"],
                    "analysis_confidence": analysis["confidence"],
                    "verification_method": analysis["verification_method"]
                })
            results.append(post)

        return results

    def convert_json_to_csv(self, json_path: str = "test_data/nside_with_persona.json",
                           csv_path: str = "test_data/nside_with_persona.csv") -> bool:
        """
        nside_with_persona.json 파일을 CSV로 변환
        
        Args:
            json_path: JSON 파일 경로
            csv_path: 변환할 CSV 파일 경로
            
        Returns:
            변환 성공 여부
        """
        print(f"🔄 {json_path}를 {csv_path}로 변환 중...")
        
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"❌ {json_path} 파일을 찾을 수 없습니다.")
            return False
            
        try:
            # JSON 파일 로드
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            # 데이터 구조 확인 및 변환
            if isinstance(data, dict):
                # 딕셔너리인 경우 리스트로 변환
                records = []
                for key, value in data.items():
                    if isinstance(value, dict):
                        record = {"id": key, **value}
                        records.append(record)
                    else:
                        records.append({"id": key, "value": value})
                data = records
            elif isinstance(data, list):
                # 이미 리스트인 경우
                records = data
            else:
                print(f"❌ 지원하지 않는 데이터 형식: {type(data)}")
                return False
                
            # DataFrame으로 변환하여 CSV로 저장
            df = pd.DataFrame(records)
            
            # CSV 파일 저장 (utf-8 인코딩 사용)
            csv_path = Path(csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, encoding='utf-8', index=False)
            
            print(f"✅ 변환 완료! {len(records)}개 항목이 {csv_path}에 저장되었습니다.")
            return True
            
        except Exception as e:
            print(f"❌ JSON to CSV 변환 실패: {e}")
            return False

    def analyze_nside_file(self, input_path: str = "test_data/nside_with_persona.csv",
                       output_path: str = "test_data/nside_analyzed.csv") -> None:
        """
        nside_with_persona.csv 파일을 로드하여 post_content 분석 후 저장

        Args:
            input_path: 입력 CSV 파일 경로
            output_path: 분석 결과 저장 CSV 경로
        """
        print("🔍 nside_with_persona.csv 분석 시작...")

        # 파일 로드
        input_path = Path(input_path)
        if not input_path.exists():
            print(f"❌ {input_path} 파일을 찾을 수 없습니다.")
            print("   JSON 파일을 CSV로 변환하시겠습니까? (y/n): ", end="")
            
            # 자동으로 JSON을 CSV로 변환 시도
            json_path = input_path.with_suffix('.json')
            if json_path.exists():
                print(f"\n🔄 {json_path}를 찾았습니다. CSV로 변환을 시도합니다...")
                if self.convert_json_to_csv(str(json_path), str(input_path)):
                    print("✅ 변환 완료! 이제 분석을 진행합니다.")
                else:
                    print("❌ 변환 실패. 수동으로 변환해주세요.")
                    return
            else:
                print("   CSV 파일이 존재하는지 확인해주세요.")
                return

        try:
            # CSV 파일을 euc-kr 인코딩으로 읽기
            df = pd.read_csv(input_path, encoding='euc-kr')
            data = df.to_dict(orient='records')
        except Exception as e:
            print(f"❌ CSV 파일 로드 실패: {e}")
            print("   파일이 CSV 형식이고 euc-kr 인코딩인지 확인해주세요.")
            return

        print(f"📊 총 {len(data)}개의 항목을 분석합니다...")

        # post_content가 있는 항목만 필터링
        posts_to_analyze = []
        for i, row in enumerate(data):
            if isinstance(row, dict) and row.get("post_content"):
                posts_to_analyze.append({"index": i, **row})

        print(f"📝 분석할 포스트: {len(posts_to_analyze)}개")

        if not posts_to_analyze:
            print("⚠️ 분석할 post_content가 없습니다.")
            return

        # 일괄 분석 실행
        analyzed_posts = self.batch_analyze_posts(posts_to_analyze)

        # 결과를 원본 데이터에 통합
        for analyzed_post in analyzed_posts:
            index = analyzed_post.pop("index")  # 임시로 추가했던 index 제거
            data[index].update(analyzed_post)

        # 분석된 데이터를 CSV로 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # DataFrame으로 변환하여 저장
        result_df = pd.DataFrame(data)
        result_df.to_csv(output_path, encoding='euc-kr', index=False)

        print(f"✅ 분석 완료! 결과가 {output_path}에 저장되었습니다.")

        # 분석 통계 출력
        self._print_analysis_stats(analyzed_posts)

    def _print_analysis_stats(self, analyzed_posts: List[Dict]) -> None:
        """분석 결과 통계 출력"""
        print("\n📊 분석 결과 통계:")
        print("-" * 50)

        # 카테고리별 분포
        categories = {}
        confidence_ranges = {"0.0-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
        verification_methods = {}

        for post in analyzed_posts:
            # 카테고리 통계
            cat = post.get("analyzed_category", "미분류")
            categories[cat] = categories.get(cat, 0) + 1

            # 신뢰도 통계
            conf = post.get("analysis_confidence", 0.0)
            if conf < 0.5:
                confidence_ranges["0.0-0.5"] += 1
            elif conf < 0.7:
                confidence_ranges["0.5-0.7"] += 1
            elif conf < 0.9:
                confidence_ranges["0.7-0.9"] += 1
            else:
                confidence_ranges["0.9-1.0"] += 1

            # 검증 방법 통계
            method = post.get("verification_method", "unknown")
            verification_methods[method] = verification_methods.get(method, 0) + 1

        print("🏷️ 카테고리별 분포:")
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat}: {count}개")

        print("\n🎯 신뢰도 분포:")
        for range_name, count in confidence_ranges.items():
            print(f"  {range_name}: {count}개")

        print("\n🔍 검증 방법 분포:")
        for method, count in sorted(verification_methods.items(), key=lambda x: x[1], reverse=True):
            print(f"  {method}: {count}개")

        print("-" * 50)


def main():
    """메인 실행 함수"""
    print("🔍 Post Content Analyzer 시작")

    # 환경 변수 확인
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("   .env 파일에 GEMINI_API_KEY를 설정해주세요.")
        return

    # 분석기 초기화
    analyzer = PostContentAnalyzer()

    # nside 파일 분석 실행 (CSV가 없으면 자동으로 JSON을 CSV로 변환)
    analyzer.analyze_nside_file()


if __name__ == "__main__":
    main()
