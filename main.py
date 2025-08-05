# from agents.input_agent import InputAgent
# from agents.plan_agent import PlanAgent
# from agents.title_agent import TitleAgent
# from agents.content_agent import ContentAgent
# from agents.image_agent import ImageAgent
# from agents.evaluation_agent import EvaluationAgent
# from agents.improve_agent import ImproveAgent
# from agents.output_agent import OutputAgent

# # 각 에이전트 초기화
# input_agent = InputAgent()
# plan_agent = PlanAgent()
# title_agent = TitleAgent()
# content_agent = ContentAgent()
# image_agent = ImageAgent()
# eval_agent = EvaluationAgent()
# improve_agent = ImproveAgent()
# output_agent = OutputAgent()

# # 실행 흐름
# data = input_agent.collect()
# steps = plan_agent.plan(data)
# title = title_agent.generate(data)
# content = content_agent.generate(data, title)
# content_with_images = image_agent.map(data, content)
# evaluation = eval_agent.evaluate(title, content_with_images)
# title, content = improve_agent.improve(title, content_with_images, evaluation)
# output_agent.export(title, content, evaluation)

from agents.input_agent import InputAgent

if __name__ == "__main__":
    input_data = InputAgent().collect()
    print(input_data)