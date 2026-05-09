import json
from unittest.mock import MagicMock, patch

from src.quiz.llm_client import LLMClient, MODEL_NAME
from src.quiz.models import (
    QuizQuestion,
    QuizSession,
    QuizType,
    TopicPerformance,
)
from src.quiz.quiz_agent import QuizAgent


class TestLLMClientMock:
    def test_mock_generate_questions(self):
        client = LLMClient(api_key=None)
        questions = client.generate_questions(
            topic="neural_networks", section="Intro", num_questions=1
        )
        assert len(questions) >= 1
        q = questions[0]
        assert "question_text" in q
        assert "options" in q

    def test_mock_evaluate_answer_mc(self):
        client = LLMClient(api_key=None)
        result = client.evaluate_answer(
            question_text="Test?",
            correct_answer="B",
            user_answer="B",
            quiz_type="multiple_choice",
        )
        assert result["is_correct"] is True
        assert result["score"] == 1.0

    def test_mock_evaluate_wrong_answer(self):
        client = LLMClient(api_key=None)
        result = client.evaluate_answer(
            question_text="Test?",
            correct_answer="B",
            user_answer="A",
            quiz_type="multiple_choice",
        )
        assert result["is_correct"] is False
        assert result["score"] == 0.0

    def test_mock_study_path(self):
        client = LLMClient(api_key=None)
        perfs = {"test": TopicPerformance(topic="test", total_questions=5, correct_count=2)}
        result = client.suggest_study_path(perfs, ["test", "other"])
        assert "weak_topics" in result
        assert "reasoning" in result


class TestQuizAgent:
    def test_start_quiz(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        session = agent.start_quiz(
            topics=["neural_networks"],
            num_questions=1,
            difficulty=3,
            focus_weak=False,
        )
        assert len(session.questions) >= 1
        assert session.is_complete is False

    def test_answer_question(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        session = agent.start_quiz(
            topics=["neural_networks"],
            num_questions=1,
            focus_weak=False,
        )
        question = session.questions[0]
        correct = question.correct_answer
        answer = agent.answer_question(correct)
        assert answer.question_id == question.id

    def test_session_summary(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        agent.start_quiz(topics=["neural_networks"], num_questions=1, focus_weak=False)
        question = agent.get_current_question()
        if question:
            agent.answer_question(question.correct_answer)
        summary = agent.get_session_summary()
        assert "score" in summary
        assert "total_questions" in summary

    def test_overall_stats(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        stats = agent.get_overall_stats()
        assert "topics_studied" in stats
        assert "total_questions" in stats
        assert "due_reviews" in stats

    def test_study_path(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        path = agent.get_study_path()
        assert hasattr(path, "weak_topics")
        assert hasattr(path, "reasoning")

    def test_resume_or_start(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        session = agent.resume_or_start(num_questions=1)
        assert len(session.questions) >= 1

    def test_list_topics(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        assert len(agent.all_topics) > 0
        assert "neural_networks" in agent.all_topics

    def test_all_topics_have_sections(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        for topic in agent.all_topics:
            assert topic in agent.topic_sections

    def test_persistence_after_answer(self, tmp_path):
        agent = QuizAgent(data_dir=str(tmp_path))
        session = agent.start_quiz(topics=["neural_networks"], num_questions=1, focus_weak=False)
        question = agent.get_current_question()
        if question:
            agent.answer_question(question.correct_answer)
        agent2 = QuizAgent(data_dir=str(tmp_path))
        assert len(agent2.performances) > 0
