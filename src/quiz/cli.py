from __future__ import annotations

import argparse
import sys

from src.quiz.models import QuizType
from src.quiz.quiz_agent import QuizAgent, TOPICS


def print_question(question) -> None:
    print(f"\n{'=' * 60}")
    print(f"Topic: {question.topic} | Section: {question.section}")
    print(f"Type: {question.quiz_type.value} | Difficulty: {question.difficulty}/5")
    print(f"{'=' * 60}")
    print(f"\n{question.question_text}\n")

    if question.quiz_type == QuizType.MULTIPLE_CHOICE:
        for opt in question.options:
            print(f"  {opt.label}) {opt.text}")
        print()
    elif question.quiz_type == QuizType.CODE_COMPLETION:
        print(f"Code template:\n{question.code_template}\n")
    elif question.quiz_type == QuizType.CONCEPT_EXPLANATION:
        print("Provide a detailed explanation in your own words.\n")


def print_feedback(answer) -> None:
    status = "CORRECT" if answer.is_correct else "INCORRECT"
    symbol = "+" if answer.is_correct else "-"
    print(f"\n  [{symbol}] {status} (Score: {answer.score:.0%})")
    if answer.feedback:
        print(f"  Feedback: {answer.feedback}")
    print()


def print_summary(agent: QuizAgent) -> None:
    summary = agent.get_session_summary()
    if not summary:
        print("No session data.")
        return
    print(f"\n{'=' * 60}")
    print("SESSION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Questions: {summary['answered']}/{summary['total_questions']}")
    print(f"  Score: {summary['score']:.0%}")
    print(f"  Status: {'Complete' if summary['is_complete'] else 'In Progress'}")
    if summary["topic_scores"]:
        print("\n  Topic Breakdown:")
        for topic, scores in summary["topic_scores"].items():
            print(f"    {topic}: {scores['accuracy']:.0%} ({scores['correct']}/{scores['total']})")
    print()


def print_stats(agent: QuizAgent) -> None:
    stats = agent.get_overall_stats()
    print(f"\n{'=' * 60}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Topics studied: {stats['topics_studied']}/{len(agent.all_topics)}")
    print(f"  Total questions answered: {stats['total_questions']}")
    print(f"  Overall accuracy: {stats['overall_accuracy']:.0%}")
    if stats["weakest_topics"]:
        print(f"  Weakest topics: {', '.join(stats['weakest_topics'])}")
    if stats["due_reviews"]:
        print(f"  Due for review: {', '.join(stats['due_reviews'])}")
    print()


def print_study_path(agent: QuizAgent) -> None:
    path = agent.get_study_path()
    print(f"\n{'=' * 60}")
    print("STUDY PATH RECOMMENDATION")
    print(f"{'=' * 60}")
    if path.weak_topics:
        print(f"  Weak topics: {', '.join(path.weak_topics)}")
    if path.suggested_sections:
        print(f"  Suggested sections: {', '.join(path.suggested_sections)}")
    if path.focus_areas:
        print(f"  Focus areas: {', '.join(path.focus_areas)}")
    if path.reasoning:
        print(f"  Reasoning: {path.reasoning}")
    print()


def interactive_quiz(agent: QuizAgent) -> None:
    print("\nDeep Learning Quiz Agent (z-ai/glm-5.1)")
    print("Commands: answer, explain, stats, study-path, quit, help")

    while True:
        question = agent.get_current_question()
        if question is None:
            print("\nQuiz session complete!")
            print_summary(agent)
            break

        print_question(question)
        user_input = input("Your answer (or command): ").strip()

        if user_input.lower() in ("quit", "q", "exit"):
            print_summary(agent)
            break
        elif user_input.lower() == "explain":
            explanation = agent.get_explanation()
            print(f"\n  Explanation: {explanation}\n")
            continue
        elif user_input.lower() == "stats":
            print_stats(agent)
            continue
        elif user_input.lower() == "study-path":
            print_study_path(agent)
            continue
        elif user_input.lower() == "help":
            print("  answer - type your answer to the current question")
            print("  explain - get an explanation for the current question")
            print("  stats - show overall statistics")
            print("  study-path - get a personalized study path recommendation")
            print("  quit - end the quiz session")
            continue
        elif not user_input:
            continue

        try:
            feedback = agent.answer_question(user_input)
            print_feedback(feedback)
        except ValueError as e:
            print(f"  Error: {e}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Learning Quiz Agent with Spaced Repetition")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI-compatible API key")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory for persistence")
    parser.add_argument("--topics", nargs="+", type=str, default=None, help="Topics to quiz on")
    parser.add_argument("--type", type=str, default="multiple_choice",
                        choices=["multiple_choice", "code_completion", "concept_explanation"],
                        help="Quiz type")
    parser.add_argument("--num-questions", type=int, default=5, help="Number of questions")
    parser.add_argument("--difficulty", type=int, default=3, choices=range(1, 6), help="Difficulty level 1-5")
    parser.add_argument("--no-focus-weak", action="store_true", help="Don't focus on weak areas")
    parser.add_argument("--stats", action="store_true", help="Show overall statistics and exit")
    parser.add_argument("--study-path", action="store_true", help="Show study path recommendation and exit")
    parser.add_argument("--list-sessions", action="store_true", help="List past quiz sessions")
    parser.add_argument("--resume", type=str, default=None, help="Resume a session by ID")
    parser.add_argument("--list-topics", action="store_true", help="List available topics")

    args = parser.parse_args()

    agent = QuizAgent(
        api_key=args.api_key,
        base_url=args.base_url,
        data_dir=args.data_dir,
    )

    if args.list_topics:
        print("\nAvailable Topics:")
        for t in TOPICS:
            print(f"  {t['topic']}: {t['section']}")
        print()
        return

    if args.stats:
        print_stats(agent)
        return

    if args.study_path:
        print_study_path(agent)
        return

    if args.list_sessions:
        sessions = agent.list_sessions()
        if not sessions:
            print("No past sessions found.")
        for s in sessions:
            status = "Complete" if s["is_complete"] else "In Progress"
            print(f"  {s['id']} | {s['created_at'][:19]} | {status} | {s['num_answers']}/{s['num_questions']}")
        return

    if args.resume:
        session = agent.load_session(args.resume)
        if session is None:
            print(f"Session {args.resume} not found.")
            return
        print(f"Resumed session {session.id} ({len(session.answers)}/{len(session.questions)} answered)")
    else:
        agent.start_quiz(
            topics=args.topics,
            quiz_type=args.type,
            num_questions=args.num_questions,
            difficulty=args.difficulty,
            focus_weak=not args.no_focus_weak,
        )

    interactive_quiz(agent)


if __name__ == "__main__":
    main()
