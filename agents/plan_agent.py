### agents/plan_agent.py
class PlanAgent:
    def __init__(self):
        pass

    def plan(self, input_data: dict) -> list:
        """
        실행할 에이전트 순서를 정의
        리턴: 에이전트 이름 리스트
        """
        return ['TitleAgent', 'ContentAgent', 'ImageAgent', 'EvaluationAgent', 'ImproveAgent', 'OutputAgent']