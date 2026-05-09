import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from src.quiz.question_bank import QuestionBank, Question, QuestionType
from src.quiz.spaced_repetition import SpacedRepetitionTracker, TopicRecord
from src.quiz.persistence import SessionPersistence
from src.quiz.llm_interface import LLMInterface

logger = logging.getLogger(__name__)


class QuizResult:
    def __init__(
        self,
        question_id: str,
        question_type: QuestionType,
        section: str,
        topic: str,
        user_answer: str,
        correct: bool,
        score: float,
        explanation: str,
        improvement_suggestion: str = "",
    ):
        self.question_id = question_id
        self.question_type = question_type
        self.section = section
        self.topic = topic
        self.user_answer = user_answer
        self.correct = correct
        self.score = score
        self.explanation = explanation
        self.improvement_suggestion = improvement_suggestion
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> Dict:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type.value,
            "section": self.section,
            "topic": self.topic,
            "user_answer": self.user_answer,
            "correct": self.correct,
            "score": self.score,
            "explanation": self.explanation,
            "improvement_suggestion": self.improvement_suggestion,
            "timestamp": self.timestamp,
        }


class QuizSession:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.results: List[QuizResult] = []
        self.created_at = datetime.utcnow().isoformat()
        self.last_activity = self.created_at

    def add_result(self, result: QuizResult):
        self.results.append(result)
        self.last_activity = datetime.utcnow().isoformat()

    def get_stats(self) -> Dict:
        if not self.results:
            return {"total": 0, "correct": 0, "accuracy": 0.0}
        total = len(self.results)
        correct = sum(1 for r in self.results if r.correct)
        avg_score = sum(r.score for r in self.results) / total
        return {
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "average_score": avg_score,
            "by_section": self._section_breakdown(),
        }

    def _section_breakdown(self) -> Dict[str, Dict]:
        sections: Dict[str, Dict[str, List]] = {}
        for r in self.results:
            if r.section not in sections:
                sections[r.section] = {"correct": [], "scores": []}
            sections[r.section]["correct"].append(r.correct)
            sections[r.section]["scores"].append(r.score)
        return {
            s: {
                "accuracy": sum(d["correct"]) / len(d["correct"]),
                "avg_score": sum(d["scores"]) / len(d["scores"]),
                "total": len(d["correct"]),
            }
            for s, d in sections.items()
        }

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_activity": self.last_activity,
            "results": [r.to_dict() for r in self.results],
            "stats": self.get_stats(),
        }


class QuizAgent:
    def __init__(
        self,
        llm_interface: Optional[LLMInterface] = None,
        storage_dir: str = ".quiz_sessions",
    ):
        self.question_bank = QuestionBank()
        self.tracker = SpacedRepetitionTracker()
        self.persistence = SessionPersistence(storage_dir)
        self.llm = llm_interface or LLMInterface()
        self.session: Optional[QuizSession] = None

        for section_key, section_data in self.question_bank.get_all_sections().items():
            for topic in section_data.get("topics", []):
                self.tracker.add_topic(topic, section_key)

    def start_session(self, session_id: Optional[str] = None) -> QuizSession:
        self.session = QuizSession(session_id)
        logger.info("Started quiz session: %s", self.session.session_id)
        return self.session

    def end_session(self) -> Optional[Dict]:
        if self.session is None:
            return None
        stats = self.session.get_stats()
        self.save_session()
        self.session = None
        return stats

    def get_next_question(
        self,
        section: Optional[str] = None,
        question_type: Optional[QuestionType] = None,
        difficulty: Optional[str] = None,
        focus_weak: bool = True,
        use_llm: bool = False,
    ) -> Optional[Question]:
        if focus_weak and not question_type and not section:
            due_topics = self.tracker.get_due_topics()
            if due_topics:
                topic_record = due_topics[0]
                section = topic_record.section
                topic = topic_record.topic
                question = self.question_bank.get_random_question(
                    section=section, topic=topic, difficulty=difficulty
                )
                if question:
                    return question

        if use_llm and self.llm.api_key:
            return self._generate_llm_question(section, question_type, difficulty)

        question = self.question_bank.get_random_question(
            section=section, qtype=question_type, difficulty=difficulty
        )
        return question

    def submit_answer(
        self,
        question_id: str,
        user_answer: str,
        use_llm_evaluation: bool = False,
    ) -> QuizResult:
        question = self.question_bank.get_question(question_id)
        if question is None:
            raise ValueError(f"Question {question_id} not found")

        if question.type == QuestionType.MULTIPLE_CHOICE:
            correct = user_answer.upper() == question.correct_answer.upper()
            score = 1.0 if correct else 0.0
            explanation = question.explanation or ""
            improvement = "" if correct else f"The correct answer is {question.correct_answer}."

        elif question.type == QuestionType.CODE_COMPLETION:
            if use_llm_evaluation and self.llm.api_key:
                result = self._evaluate_with_llm(question, user_answer)
                correct = result.get("is_correct", False)
                score = float(result.get("score", 0.0 if not correct else 1.0))
                explanation = result.get("explanation", question.explanation or "")
                improvement = result.get("improvement_suggestion", "")
            else:
                correct = user_answer.strip().lower() == question.correct_answer.strip().lower()
                score = 1.0 if correct else 0.0
                explanation = question.explanation or ""
                improvement = "" if correct else f"Expected: {question.correct_answer}"

        elif question.type == QuestionType.CONCEPT_EXPLANATION:
            if use_llm_evaluation and self.llm.api_key:
                result = self._evaluate_with_llm(question, user_answer)
                correct = result.get("is_correct", False)
                score = float(result.get("score", 0.0))
                explanation = result.get("explanation", "")
                improvement = result.get("improvement_suggestion", "")
            else:
                key_concepts = question.key_concepts or []
                mentioned = sum(1 for concept in key_concepts if concept.lower() in user_answer.lower())
                score = mentioned / max(len(key_concepts), 1)
                correct = score >= 0.5
                explanation = f"You covered {mentioned}/{len(key_concepts)} key concepts: {', '.join(key_concepts)}"
                improvement = "" if correct else f"Try to address: {', '.join(key_concepts)}"
        else:
            correct = False
            score = 0.0
            explanation = "Unknown question type"
            improvement = ""

        quiz_result = QuizResult(
            question_id=question_id,
            question_type=question.type,
            section=question.section,
            topic=question.topic,
            user_answer=user_answer,
            correct=correct,
            score=score,
            explanation=explanation,
            improvement_suggestion=improvement,
        )

        if self.session:
            self.session.add_result(quiz_result)

        self.tracker.record_result(question.topic, question.section, correct)

        return quiz_result

    def get_study_recommendations(self) -> Dict[str, Any]:
        weak_topics = self.tracker.get_weak_topics()
        strong_topics = self.tracker.get_strong_topics()
        weak_names = [f"{r.section}/{r.topic}" for r in weak_topics]
        strong_names = [f"{r.section}/{r.topic}" for r in strong_topics]

        recommendations = {
            "weak_topics": weak_names,
            "strong_topics": strong_names,
            "due_topics": [f"{r.section}/{r.topic}" for r in self.tracker.get_due_topics()],
            "overall_stats": self.tracker.get_overall_stats(),
        }

        if self.llm.api_key and weak_names:
            try:
                study_path = self.llm.suggest_study_path(weak_names, strong_names)
                recommendations["study_path"] = study_path
            except Exception as e:
                logger.warning("Failed to generate LLM study path: %s", e)
                recommendations["study_path"] = self._generate_default_study_path(weak_topics, strong_topics)

        if "study_path" not in recommendations:
            recommendations["study_path"] = self._generate_default_study_path(weak_topics, strong_topics)

        return recommendations

    def get_explanation(self, topic: str, section: str, concept: Optional[str] = None) -> Dict:
        concept = concept or topic
        if self.llm.api_key:
            try:
                result = self.llm.generate_explanation(section, concept)
                if "parse_error" not in result:
                    return result
            except Exception as e:
                logger.warning("LLM explanation failed: %s", e)

        section_data = self.question_bank.get_all_sections().get(section, {})
        return {
            "explanation": f"Topic: {topic} in {section_data.get('title', section)}",
            "key_points": ["Review the course material for this topic"],
            "related_topics": self.question_bank.get_section_topics(section)[:5],
        }

    def save_session(self):
        if self.session is None:
            return
        data = {
            "session": self.session.to_dict(),
            "tracker": self.tracker.to_dict(),
        }
        self.persistence.save_session(self.session.session_id, data)
        logger.info("Saved quiz session: %s", self.session.session_id)

    def load_session(self, session_id: str) -> Optional[QuizSession]:
        data = self.persistence.load_session(session_id)
        if data is None:
            return None

        session_data = data.get("session", {})
        tracker_data = data.get("tracker", {})

        self.session = QuizSession(session_id=session_data.get("session_id", session_id))
        self.session.created_at = session_data.get("created_at", self.session.created_at)
        self.session.last_activity = session_data.get("last_activity", self.session.last_activity)

        for result_data in session_data.get("results", []):
            result = QuizResult(
                question_id=result_data["question_id"],
                question_type=QuestionType(result_data["question_type"]),
                section=result_data["section"],
                topic=result_data["topic"],
                user_answer=result_data["user_answer"],
                correct=result_data["correct"],
                score=result_data["score"],
                explanation=result_data.get("explanation", ""),
                improvement_suggestion=result_data.get("improvement_suggestion", ""),
            )
            result.timestamp = result_data.get("timestamp", result.timestamp)
            self.session.results.append(result)

        if tracker_data:
            self.tracker = SpacedRepetitionTracker.from_dict(tracker_data)

        return self.session

    def _generate_llm_question(
        self,
        section: Optional[str],
        question_type: Optional[QuestionType],
        difficulty: Optional[str],
    ) -> Optional[Question]:
        topic = "deep learning"
        section_data = self.question_bank.get_all_sections().get(section, {})
        if section_data:
            import random
            topic = random.choice(section_data.get("topics", [topic]))

        qtype_str = question_type.value if question_type else "multiple_choice"
        difficulty = difficulty or "medium"

        try:
            result = self.llm.generate_quiz_question(topic, qtype_str, difficulty)
            if "parse_error" in result:
                return None

            question_id = f"llm-{uuid.uuid4().hex[:8]}"
            q_type = QuestionType(result.get("type", qtype_str))

            return Question(
                id=question_id,
                type=q_type,
                section=section or "foundations",
                topic=topic,
                difficulty=difficulty,
                question_text=result.get("question", ""),
                options=result.get("options"),
                correct_answer=result.get("correct_answer"),
                explanation=result.get("explanation", ""),
                code_template=result.get("code_template"),
                blank_description=result.get("blank_description"),
                key_concepts=result.get("key_concepts"),
                grading_rubric=result.get("grading_rubric"),
                sample_answer=result.get("sample_answer"),
            )
        except Exception as e:
            logger.error("LLM question generation failed: %s", e)
            return None

    def _evaluate_with_llm(self, question: Question, user_answer: str) -> Dict:
        correct_answer = question.correct_answer or question.sample_answer
        try:
            result = self.llm.evaluate_answer(
                question=question.question_text,
                user_answer=user_answer,
                correct_answer=correct_answer,
            )
            return result
        except Exception as e:
            logger.error("LLM evaluation failed: %s", e)
            return {"is_correct": False, "score": 0.0, "explanation": str(e), "improvement_suggestion": ""}

    def _generate_default_study_path(
        self, weak_topics: List[TopicRecord], strong_topics: List[TopicRecord]
    ) -> Dict:
        path = []
        for record in weak_topics[:5]:
            path.append({
                "topic": record.topic,
                "reason": f"Accuracy: {record.correct_count}/{record.total_attempts} ({record.correct_count / max(record.total_attempts, 1) * 100:.0f}%)",
            })
        return {
            "study_path": path,
            "priority_order": [r.topic for r in weak_topics],
            "estimated_study_time": f"{len(weak_topics) * 15} minutes",
        }