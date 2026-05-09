from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class QuizType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    CODE_COMPLETION = "code_completion"
    CONCEPT_EXPLANATION = "concept_explanation"


@dataclass
class QuizOption:
    label: str
    text: str
    is_correct: bool = False


@dataclass
class QuizQuestion:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    quiz_type: QuizType = QuizType.MULTIPLE_CHOICE
    topic: str = ""
    section: str = ""
    question_text: str = ""
    options: list[QuizOption] = field(default_factory=list)
    correct_answer: str = ""
    code_template: str = ""
    explanation: str = ""
    difficulty: int = 3
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "quiz_type": self.quiz_type.value,
            "topic": self.topic,
            "section": self.section,
            "question_text": self.question_text,
            "options": [
                {"label": o.label, "text": o.text, "is_correct": o.is_correct}
                for o in self.options
            ],
            "correct_answer": self.correct_answer,
            "code_template": self.code_template,
            "explanation": self.explanation,
            "difficulty": self.difficulty,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuizQuestion:
        options = [
            QuizOption(
                label=o["label"], text=o["text"], is_correct=o.get("is_correct", False)
            )
            for o in data.get("options", [])
        ]
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            quiz_type=QuizType(data.get("quiz_type", "multiple_choice")),
            topic=data.get("topic", ""),
            section=data.get("section", ""),
            question_text=data.get("question_text", ""),
            options=options,
            correct_answer=data.get("correct_answer", ""),
            code_template=data.get("code_template", ""),
            explanation=data.get("explanation", ""),
            difficulty=data.get("difficulty", 3),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class QuizAnswer:
    question_id: str = ""
    user_answer: str = ""
    is_correct: bool = False
    feedback: str = ""
    score: float = 0.0
    answered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "user_answer": self.user_answer,
            "is_correct": self.is_correct,
            "feedback": self.feedback,
            "score": self.score,
            "answered_at": self.answered_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuizAnswer:
        return cls(
            question_id=data.get("question_id", ""),
            user_answer=data.get("user_answer", ""),
            is_correct=data.get("is_correct", False),
            feedback=data.get("feedback", ""),
            score=data.get("score", 0.0),
            answered_at=data.get(
                "answered_at", datetime.now(timezone.utc).isoformat()
            ),
        )


@dataclass
class TopicPerformance:
    topic: str = ""
    total_questions: int = 0
    correct_count: int = 0
    streak: int = 0
    easiness_factor: float = 2.5
    interval_days: int = 1
    repetitions: int = 0
    next_review: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_reviewed: str = ""

    @property
    def accuracy(self) -> float:
        if self.total_questions == 0:
            return 0.0
        return self.correct_count / self.total_questions

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "total_questions": self.total_questions,
            "correct_count": self.correct_count,
            "streak": self.streak,
            "easiness_factor": self.easiness_factor,
            "interval_days": self.interval_days,
            "repetitions": self.repetitions,
            "next_review": self.next_review,
            "last_reviewed": self.last_reviewed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TopicPerformance:
        return cls(
            topic=data.get("topic", ""),
            total_questions=data.get("total_questions", 0),
            correct_count=data.get("correct_count", 0),
            streak=data.get("streak", 0),
            easiness_factor=data.get("easiness_factor", 2.5),
            interval_days=data.get("interval_days", 1),
            repetitions=data.get("repetitions", 0),
            next_review=data.get(
                "next_review", datetime.now(timezone.utc).isoformat()
            ),
            last_reviewed=data.get("last_reviewed", ""),
        )


@dataclass
class StudyPathSuggestion:
    weak_topics: list[str] = field(default_factory=list)
    suggested_sections: list[str] = field(default_factory=list)
    focus_areas: list[str] = field(default_factory=list)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "weak_topics": self.weak_topics,
            "suggested_sections": self.suggested_sections,
            "focus_areas": self.focus_areas,
            "reasoning": self.reasoning,
        }


@dataclass
class QuizSession:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    questions: list[QuizQuestion] = field(default_factory=list)
    answers: list[QuizAnswer] = field(default_factory=list)
    topic_performances: dict[str, TopicPerformance] = field(default_factory=dict)
    current_question_index: int = 0
    is_complete: bool = False

    @property
    def score(self) -> float:
        if not self.answers:
            return 0.0
        return sum(a.score for a in self.answers) / len(self.answers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "questions": [q.to_dict() for q in self.questions],
            "answers": [a.to_dict() for a in self.answers],
            "topic_performances": {
                k: v.to_dict() for k, v in self.topic_performances.items()
            },
            "current_question_index": self.current_question_index,
            "is_complete": self.is_complete,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QuizSession:
        questions = [QuizQuestion.from_dict(q) for q in data.get("questions", [])]
        answers = [QuizAnswer.from_dict(a) for a in data.get("answers", [])]
        topic_performances = {
            k: TopicPerformance.from_dict(v)
            for k, v in data.get("topic_performances", {}).items()
        }
        return cls(
            id=data.get("id", uuid.uuid4().hex[:12]),
            created_at=data.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            questions=questions,
            answers=answers,
            topic_performances=topic_performances,
            current_question_index=data.get("current_question_index", 0),
            is_complete=data.get("is_complete", False),
        )
