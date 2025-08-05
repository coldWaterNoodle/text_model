### agents/content_agent.py
class ContentAgent:
    def __init__(self):
        pass

    def generate(self, input_data: dict, title: str) -> str:
        """
        본문 생성
        """
        return f"# {title}\n\n본문 내용 예시입니다."