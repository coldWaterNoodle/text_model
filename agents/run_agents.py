import os
import argparse
import json
from input_agent import InputAgent
from plan_agent import PlanAgent
from title_agent import TitleAgent
from content_agent import ContentAgent
# from image_agent import ImageAgent
# from eval_agent import EvalAgent
# from final_assembler import FinalAssembler


def load_env(env_path: str = ".env"):
    """
    Load environment variables from .env file.
    """
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=env_path)


def parse_args():
    """
    Parse command-line arguments:
      mode: 'test' or 'use' (default: 'use')
      test_case_num: numeric test case for test mode (예: 11)
      --env: path to environment file
    """
    parser = argparse.ArgumentParser(description="블로그 자동화 멀티 에이전트 실행기")
    parser.add_argument(
        'mode', nargs='?', choices=['test', 'use'], default='use',
        help="모드 선택: 'test' 또는 'use' (기본: 'use')"
    )
    parser.add_argument(
        'test_case_num', nargs='?',
        help="테스트 모드일 때 사용할 케이스 번호 (예: 11)"
    )
    parser.add_argument(
        '--env', default=".env",
        help="환경변수 파일 경로 (기본: .env)"
    )
    return parser.parse_args()


def main():
    # 1. 인자 파싱 및 환경변수 로드
    args = parse_args()
    load_env(args.env)

    # 2. Input 데이터 준비
    agent = InputAgent()
    if args.mode == "test":
        input_data = agent.run_test()
    else:  # use 모드
        input_data = agent.run_use()
    
    if input_data is None:
        print("입력 데이터 수집에 실패했습니다.")
        return

    # test 모드일 때 터미널 출력
    if args.mode == 'test':
        print(f"🔍 [TEST MODE] using case: test_case_{args.test_case_num or '1'}")
        print("🔍 [INPUT DATA]", json.dumps(input_data, indent=2, ensure_ascii=False))

    # 3. PlanAgent 실행
    #    - test 모드: 2-way로 고정 (rounds=2)
    #    - use 모드 : CORT 모드, PlanAgent 내부 default_nway_rounds 사용 (rounds=None)
    plan_agent = PlanAgent()
    plan_rounds = 2 if args.mode == 'test' else None
    plan, plan_candidates, plan_eval_info, _ = plan_agent.generate(
        input_data=input_data,
        mode='cli',
        rounds=plan_rounds
    )
    plan_agent.save_log(
        input_data=input_data,
        candidates=plan_candidates,
        best_output=json.dumps(plan, ensure_ascii=False),
        selected=plan_eval_info['selected'],
        reason=plan_eval_info['reason'],
        mode=args.mode
    )

    # test 모드일 때 PlanAgent 결과 출력
    if args.mode == 'test':
        print("🔍 [PLAN CANDIDATES]")
        print(json.dumps(plan_candidates, indent=2, ensure_ascii=False))
        print(f"🔍 [PLAN SELECTED] {plan_eval_info['selected']}")
        print("🔍 [PLAN REASON]", json.dumps(plan_eval_info['reason'], indent=2, ensure_ascii=False))
        print("🔍 [PLAN RESULT]", json.dumps(plan, indent=2, ensure_ascii=False))

    # 4. TitleAgent 실행
    title_agent = TitleAgent()
    # test 모드: 2-way, use 모드: CORT n-way (internal default)
    title_rounds = 2 if args.mode == 'test' else None
    title, title_candidates, title_eval_info, _ = title_agent.generate(
        input_data=plan,
        mode='cli',
        rounds=title_rounds
    )
    title_agent.save_log(
        input_data=plan,
        candidates=title_candidates,
        best_output=json.dumps(title, ensure_ascii=False),
        selected=title_eval_info['selected'],
        reason=title_eval_info['reason'],
        mode=args.mode
    )

    # test 모드일 때 TitleAgent 결과 출력
    if args.mode == 'test':
        print("🔍 [TITLE CANDIDATES]")
        print(json.dumps(title_candidates, indent=2, ensure_ascii=False))
        print(f"🔍 [TITLE SELECTED] {title_eval_info['selected']}")
        print("🔍 [TITLE REASON]", json.dumps(title_eval_info['reason'], indent=2, ensure_ascii=False))
        print("🔍 [TITLE RESULT]", json.dumps(title, indent=2, ensure_ascii=False))

    # 5. ContentAgent 실행
    content_agent = ContentAgent()
    content, content_candidates, content_eval_info, _ = content_agent.generate(
        input_data={**input_data, **plan, 'title': title},
        mode=args.mode
    )
    content_agent.save_log(
        input_data={**input_data, **plan, 'title': title},
        candidates=content_candidates,
        best_output=content,
        selected=content_eval_info['selected'],
        reason=content_eval_info['reason'],
        mode=args.mode
    )

    # test 모드일 때 ContentAgent 결과 출력
    if args.mode == 'test':
        print("🔍 [CONTENT CANDIDATES]")
        print(json.dumps(content_candidates, indent=2, ensure_ascii=False))
        print(f"🔍 [CONTENT SELECTED] {content_eval_info['selected']}")
        print("🔍 [CONTENT REASON]", json.dumps(content_eval_info['reason'], indent=2, ensure_ascii=False))
        print("🔍 [CONTENT RESULT]", json.dumps(content, indent=2, ensure_ascii=False))

    # 전체 글 출력 (제목 + 내용)
    full_article = content_agent.format_full_article(content, input_data={**input_data, **plan, 'title': title})
    print("\n" + "="*80)
    print("📝 [FULL ARTICLE]")
    print("="*80)
    print(full_article)
    print("="*80)

    # image_agent = ImageAgent()
    # images, image_candidates, image_eval_info, _ = image_agent.generate(content)

    # eval_agent = EvalAgent()
    # quality_report = eval_agent.evaluate(plan, title, content, images)

    # assembler = FinalAssembler()
    # final_output = assembler.assemble(
    #     plan=plan,
    #     title=title,
    #     content=content,
    #     images=images,
    #     evaluation=quality_report
    # )

    # if args.mode == 'test':
    #     print("🔍 [FINAL OUTPUT]", json.dumps(final_output, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()