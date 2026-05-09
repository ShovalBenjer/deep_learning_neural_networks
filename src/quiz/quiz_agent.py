from __future__ import annotations

from typing import Any

from src.quiz.llm_client import LLMClient
from src.quiz.models import (
    QuizAnswer,
    QuizQuestion,
    QuizSession,
    QuizType,
    StudyPathSuggestion,
    TopicPerformance,
)
from src.quiz.persistence import PersistenceManager
from src.quiz.spaced_repetition import SpacedRepetitionScheduler


TOPICS: list[dict[str, str]] = [
    {"topic": "neural_networks", "section": "Introduction to Artificial Neural Networks"},
    {"topic": "perceptron", "section": "The Perceptron: A Basic Neural Unit"},
    {"topic": "activation_functions", "section": "Neuron Activation Functions: Introducing Non-Linearity"},
    {"topic": "learning", "section": "Learning in Neural Networks"},
    {"topic": "gradient_descent", "section": "Gradient Descent: Optimizing Network Weights"},
    {"topic": "backpropagation", "section": "Error Backpropagation: Computing Gradients in Deep Networks"},
    {"topic": "cost_functions", "section": "Cost Functions: Measuring Network Performance"},
    {"topic": "overfitting", "section": "Overfitting and Regularization: Enhancing Generalization"},
    {"topic": "vanishing_gradients", "section": "Vanishing and Exploding Gradients: Deep Network Challenges"},
    {"topic": "local_minima", "section": "Addressing Local Minima and Optimization Challenges"},
    {"topic": "deep_architectures", "section": "Deep Architectures and Layer-wise Training"},
    {"topic": "cnns", "section": "Convolutional Neural Networks (CNNs): Image and Spatial Data"},
    {"topic": "rnns", "section": "Recurrent Neural Networks (RNNs): Sequence Data and Time Series"},
    {"topic": "lstms", "section": "Long Short-Term Memory Networks (LSTMs) and GRUs"},
    {"topic": "attention", "section": "Attention Mechanisms and Transformers"},
    {"topic": "evaluation", "section": "Model Evaluation and Performance Metrics"},
]


class QuizAgent:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        data_dir: str | None = None,
    ):
        self.llm = LLMClient(api_key=api_key, base_url=base_url)
        self.scheduler = SpacedRepetitionScheduler()
        self.persistence = PersistenceManager(data_dir=data_dir)
        self.performances: dict[str, TopicPerformance] = self.persistence.load_performance()
        self._session: QuizSession | None = None

    @property
    def all_topics(self) -> list[str]:
        return [t["topic"] for t in TOPICS]

    @property
    def topic_sections(self) -> dict[str, str]:
        return {t["topic"]: t["section"] for t in TOPICS}

    def start_quiz(
        self,
        topics: list[str] | None = None,
        quiz_type: str = "multiple_choice",
        num_questions: int = 5,
        difficulty: int = 3,
        focus_weak: bool = True,
    ) -> QuizSession:
        valid_types = {t.value for t in QuizType}
        if quiz_type not in valid_types:
            raise ValueError(f"Invalid quiz_type '{quiz_type}'. Must be one of: {sorted(valid_types)}")

        if topics is None:
            if focus_weak:
                topics = self.scheduler.get_priority_topics(
                    self.performances, self.all_topics
                )[:5]
            else:
                topics = self.all_topics[:5]

        questions: list[QuizQuestion] = []
        per_topic = max(1, num_questions // len(topics))
        remaining = num_questions

        for topic in topics:
            n = min(per_topic, remaining)
            if n <= 0:
                break
            section = self.topic_sections.get(topic, topic)
            raw_questions = self.llm.generate_questions(
                topic=topic,
                section=section,
                quiz_type=quiz_type,
                num_questions=n,
                difficulty=difficulty,
            )
            for rq in raw_questions:
                questions.append(QuizQuestion.from_dict(rq))
            remaining -= len(raw_questions)

        if remaining > 0 and topics:
            for i, topic in enumerate(topics):
                if remaining <= 0:
                    break
                n = min(1, remaining)
                section = self.topic_sections.get(topic, topic)
                raw_questions = self.llm.generate_questions(
                    topic=topic,
                    section=section,
                    quiz_type=quiz_type,
                    num_questions=n,
                    difficulty=difficulty,
                )
                for rq in raw_questions:
                    questions.append(QuizQuestion.from_dict(rq))
                remaining -= len(raw_questions)

        session = QuizSession(questions=questions)
        self._session = session
        try:
            self.persistence.save_session(session)
        except OSError:
            pass
        return session

    def get_current_question(self) -> QuizQuestion | None:
        if self._session is None:
            return None
        if self._session.current_question_index >= len(self._session.questions):
            return None
        return self._session.questions[self._session.current_question_index]

    def answer_question(self, answer: str) -> QuizAnswer:
        if self._session is None:
            raise ValueError("No active quiz session. Call start_quiz() first.")

        question = self.get_current_question()
        if question is None:
            raise ValueError("No more questions in this session.")

        evaluation = self.llm.evaluate_answer(
            question_text=question.question_text,
            correct_answer=question.correct_answer,
            user_answer=answer,
            quiz_type=question.quiz_type.value,
        )

        quiz_answer = QuizAnswer(
            question_id=question.id,
            user_answer=answer,
            is_correct=evaluation.get("is_correct", False),
            feedback=evaluation.get("feedback", ""),
            score=evaluation.get("score", 0.0),
        )

        self._session.answers.append(quiz_answer)
        self._session.current_question_index += 1

        topic = question.topic
        if topic not in self._session.topic_performances:
            self._session.topic_performances[topic] = TopicPerformance(topic=topic)

        perf = self._session.topic_performances[topic]
        quality = SpacedRepetitionScheduler.quality_from_score(quiz_answer.score)
        self.scheduler.update_performance(perf, quality)

        if topic not in self.performances:
            self.performances[topic] = TopicPerformance(topic=topic)
        global_perf = self.performances[topic]
        self.scheduler.update_performance(global_perf, quality)

        if self._session.current_question_index >= len(self._session.questions):
            self._session.is_complete = True

        try:
            self.persistence.save_session(self._session)
            self.persistence.save_performance(self.performances)
        except OSError:
            pass

        return quiz_answer

    def get_session_summary(self) -> dict[str, Any]:
        if self._session is None:
            return {}
        session = self._session
        return {
            "session_id": session.id,
            "total_questions": len(session.questions),
            "answered": len(session.answers),
            "score": session.score,
            "is_complete": session.is_complete,
            "topic_scores": {
                topic: {
                    "accuracy": perf.accuracy,
                    "total": perf.total_questions,
                    "correct": perf.correct_count,
                }
                for topic, perf in session.topic_performances.items()
            },
        }

    def get_study_path(self) -> StudyPathSuggestion:
        result = self.llm.suggest_study_path(self.performances, self.all_topics)
        return StudyPathSuggestion(
            weak_topics=result.get("weak_topics", []),
            suggested_sections=result.get("suggested_sections", []),
            focus_areas=result.get("focus_areas", []),
            reasoning=result.get("reasoning", ""),
        )

    def get_explanation(self, question_id: str | None = None) -> str:
        if self._session is None:
            return "No active session."
        question = None
        if question_id:
            for q in self._session.questions:
                if q.id == question_id:
                    question = q
                    break
        if question is None:
            question = self.get_current_question()
        if question is None:
            return "No question found."
        return self.llm.generate_explanation(
            topic=question.topic,
            question_text=question.question_text,
            correct_answer=question.correct_answer,
        )

    def load_session(self, session_id: str) -> QuizSession | None:
        session = self.persistence.load_session(session_id)
        if session:
            self._session = session
        return session

    def list_sessions(self) -> list[dict]:
        return self.persistence.list_sessions()

    def get_overall_stats(self) -> dict[str, Any]:
        if not self.performances:
            return {
                "topics_studied": 0,
                "total_questions": 0,
                "overall_accuracy": 0.0,
                "weakest_topics": [],
                "due_reviews": self.all_topics,
            }
        total_q = sum(p.total_questions for p in self.performances.values())
        total_correct = sum(p.correct_count for p in self.performances.values())
        due = self.scheduler.get_due_topics(self.performances)
        weak = self.scheduler.get_weak_topics(self.performances)
        return {
            "topics_studied": len(self.performances),
            "total_questions": total_q,
            "overall_accuracy": total_correct / total_q if total_q > 0 else 0.0,
            "weakest_topics": weak[:5],
            "due_reviews": due,
        }

    def resume_or_start(self, quiz_type: str = "multiple_choice", num_questions: int = 5) -> QuizSession:
        if self._session and not self._session.is_complete:
            return self._session
        return self.start_quiz(quiz_type=quiz_type, num_questions=num_questions)
