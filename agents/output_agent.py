### agents/output_agent.py
class OutputAgent:
    def __init__(self):
        pass

    def export(self, title: str, content: str, evaluation: dict):
        """
        최종 결과 저장 또는 블로그 업로드
        """
        print("제목:", title)
        print("내용:\n", content)
        print("평가:", evaluation)