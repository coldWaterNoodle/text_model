import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from google.generativeai import GenerativeModel, configure
from dotenv import load_dotenv

# 🔧 환경 및 모델 설정
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("환경변수 GEMINI_API_KEY가 설정되지 않았습니다.")
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel(model_name="models/gemini-1.5-flash")

class ContentAgent:
    BLOG_STRUCTURE = [
        ("intro", "content1_intro_prompt.txt"),
        ("visit_reason", "content2_visit_prompt.txt"),
        ("inspection", "content3_inspection_prompt.txt"),
        ("doctor_tip", "content4_doctor_tip_prompt.txt"),
        ("treatment", "content5_treatment_prompt.txt"),
        ("check_point", "content6_check_point_prompt.txt"),
        ("conclusion", "content7_conclusion_prompt.txt")
    ]

    def __init__(
        self, 
        prompt_dir: str = "test_prompt",
        eval2_template_path: str = "test_prompt/content_evaluation_prompt.txt",
        nway_template_path: str = "test_prompt/content_nway_evaluation_prompt.txt",
        default_nway_rounds: int = 3
    ):
        self.prompt_dir = Path(prompt_dir)
        self.prompts = self._load_prompts()
        self.eval2_template = self._load_template(eval2_template_path, "2-way 평가")
        self.nway_template = self._load_template(nway_template_path, "N-way 평가")
        self.default_nway_rounds = default_nway_rounds

    def _load_prompts(self) -> Dict[str, str]:
        prompts: Dict[str, str] = {}
        for section, filename in self.BLOG_STRUCTURE:
            path = self.prompt_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {path}")
            prompts[section] = path.read_text(encoding="utf-8")
        return prompts

    def _load_template(self, path: str, name: str) -> str:
        file = Path(path)
        if not file.exists():
            raise FileNotFoundError(f"{name} 프롬프트 파일을 찾을 수 없습니다: {file}")
        return file.read_text(encoding="utf-8")

    def generate(
        self,
        input_data: dict,
        mode: str = "use",
        rounds: Optional[int] = None
    ) -> Tuple[str, Dict[str, str], Dict, dict]:
        """
        Returns:
        - best_output: 선택된 최종 콘텐츠
        - candidates: 모든 후보들
        - evaluation_info: 선택 정보와 이유
        - input_data: 입력 데이터
        """
        # Determine number of candidates
        if rounds is None:
            rounds = 2 if mode == "test" else self.default_nway_rounds

        # Generate multiple content candidates
        candidates = self._generate_candidates(input_data, rounds)
        candidates_dict = {f"후보 {i+1}": candidates[i] for i in range(len(candidates))}

        # Evaluate candidates
        if rounds == 2:
            best_output, selected, reason = self._eval_2way(candidates)
        else:
            best_output, selected, reason = self._eval_nway(candidates)

        evaluation_info = {"selected": selected, "reason": reason}
        return best_output, candidates_dict, evaluation_info, input_data

    def _generate_candidates(self, input_data: dict, rounds: int) -> List[str]:
        """Generate multiple content candidates by generating each section multiple times"""
        candidates: List[str] = []
        
        for i in range(rounds):
            print(f"\n🔄 [후보 {i+1} 생성 중...]")
            blog_sections = []
            
            # 각 섹션별로 생성
            for section, _ in self.BLOG_STRUCTURE:
                section_content = self.generate_section(section, input_data)
                blog_sections.append(section_content)
            
            # 전체 콘텐츠 조합
            full_content = "\n\n".join(blog_sections)
            
            if full_content and full_content not in candidates:
                candidates.append(full_content)
                print(f"✅ [후보 {i+1}] 생성 완료")
            else:
                print(f"⚠️ [후보 {i+1}] 빈 응답 또는 중복")
        
        if len(candidates) < 2:
            raise ValueError(f"콘텐츠 후보가 부족합니다: {len(candidates)}")
        return candidates[:rounds]

    def _generate_content(self, context_dict: Dict) -> str:
        """기존 generate 메서드의 로직을 여기로 이동"""
        blog_sections = []
        for section, _ in self.BLOG_STRUCTURE:
            blog_sections.append(self.generate_section(section, context_dict))
        return "\n\n".join(blog_sections)

    def generate_section(self, section: str, context_dict: Dict) -> str:
        prompt = self.prompts[section] % context_dict
        # Note: using old-style formatting to avoid braces conflicts
        print(f"\n🔍 [{section}] 프롬프트:\n{prompt}\n")
        response = model.generate_content(prompt)
        output = response.text.strip()
        print(f"\n✅ [{section}] 생성 결과:\n{output}\n")
        return output

    def _eval_2way(self, candidates: List[str]) -> Tuple[str, str, str]:
        """2-way evaluation for content candidates"""
        prompt = self.eval2_template.format(
            candidate_1=candidates[0],
            candidate_2=candidates[1]
        )
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        try:
            doc = json.loads(txt)
            sel = doc.get("selected", "").strip()
            reason = doc.get("reason", "").strip()
        except:
            m = re.search(r"후보\s*([12])", txt)
            sel = f"후보 {m.group(1)}" if m else "후보 1"
            reason = txt
        idx = int(sel.replace("후보", "")) - 1
        return candidates[idx], sel, reason

    def _eval_nway(self, candidates: List[str]) -> Tuple[str, str, str]:
        """N-way evaluation for content candidates"""
        # Build block listing for all candidates
        blocks = "\n".join(f"후보 {i+1}:\n{cand}" for i, cand in enumerate(candidates))
        
        # Prepare format args including count and block
        fmt_args = {'n': len(candidates), 'candidates': blocks}
        # Add individual placeholders for safety
        for i, c in enumerate(candidates, start=1):
            fmt_args[f'candidate_{i}'] = c
        
        prompt = self.nway_template.format(**fmt_args)
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        try:
            doc = json.loads(txt)
            sel = doc.get("selected", "").strip()
            reason = doc.get("reason", "").strip()
        except:
            m = re.search(r"후보\s*(\d+)", txt)
            num = m.group(1) if m else "1"
            sel = f"후보 {num}"
            reason = txt
        idx = int(sel.replace("후보", "")) - 1
        return candidates[idx], sel, reason

    def format_full_article(self, content: str, input_data: dict = None, title: str = None) -> str:
        """제목과 내용을 합쳐서 전체 글을 포맷팅"""
        # title을 가져오는 방법: 1) title 파라미터, 2) input_data에서, 3) 기본값
        if title is None:
            if input_data and 'title' in input_data:
                title = input_data['title']
            else:
                title = "제목 없음"
        
        # title이 문자열이 아닌 경우 문자열로 변환
        if not isinstance(title, str):
            title = str(title)
        
        # 제목에서 줄바꿈 기호 제거하고 정리
        clean_title = title.replace('\n', ' ').strip()
        
        # 전체 글 구성
        full_article = f"{clean_title}\n\n{content}"
        return full_article

    def save_log(
        self,
        input_data: dict,
        candidates: Dict[str, str],
        best_output: str,
        selected: str,
        reason: str,
        mode: str = "use"
    ) -> None:
        """
        mode='test' -> 저장 in test_logs/test
        mode='use'  -> 저장 in test_logs/use
        """
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = Path(f"test_logs/{mode}")
        dir_path.mkdir(parents=True, exist_ok=True)
        with open(dir_path / f"{now}_content_log.json", "w", encoding="utf-8") as f:
            json.dump({
                "input": input_data,
                "candidates": candidates,
                "selected": selected,
                "reason": reason,
                "best_output": best_output
            }, f, ensure_ascii=False, indent=2)

# CLI 테스트
if __name__ == "__main__":
    print("🔍 ContentAgent CLI 테스트 시작")
    agent = ContentAgent()
    
    # 테스트용 입력 데이터 (실제로는 run_agents.py에서 전달받음)
    test_input = {
        "patient_name": "김환자",
        "age": "35",
        "gender": "여성",
        "symptoms": "치통",
        "diagnosis": "충치",
        "treatment": "충치 치료",
        "title": "35세 여성 환자의 치통 치료 경험"
    }
    
    # 콘텐츠 생성
    content, candidates, eval_info, _ = agent.generate(test_input, mode="test")
    
    # 전체 글 출력
    full_article = agent.format_full_article(content, test_input)
    print("\n" + "="*80)
    print("📝 [FULL ARTICLE]")
    print("="*80)
    print(full_article)
    print("="*80)
