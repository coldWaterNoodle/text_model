### agents/image_agent.py
class ImageAgent:
    def __init__(self):
        pass

    def map(self, input_data: dict, content: str) -> str:
        """
        이미지 파일명을 본문에 삽입
        """
        return content + "\n\n![예시 이미지](경로/파일명.png)"