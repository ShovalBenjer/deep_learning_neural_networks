import unittest
import tempfile
import shutil

from src.quiz.quiz_agent import QuizAgent, QuizSession, QuizResult
from src.quiz.question_bank import QuestionType


class TestQuizSession(unittest.TestCase):
    def test_create_session(self):
        session = QuizSession()
        self.assertIsNotNone(session.session_id)
        self.assertEqual(len(session.results), 0)

    def test_create_session_with_id(self):
        session = QuizSession(session_id="test-id")
        self.assertEqual(session.session_id, "test-id")

    def test_add_result(self):
        session = QuizSession()
        result = QuizResult(
            question_id="q1",
            question_type=QuestionType.MULTIPLE_CHOICE,
            section="foundations",
            topic="ReLU",
            user_answer="A",
            correct=True,
            score=1.0,
            explanation="Correct!",
        )
        session.add_result(result)
        self.assertEqual(len(session.results), 1)

    def test_get_stats_empty(self):
        session = QuizSession()
        stats = session.get_stats()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["accuracy"], 0.0)

    def test_get_stats_with_results(self):
        session = QuizSession()
        session.add_result(QuizResult("q1", QuestionType.MULTIPLE_CHOICE, "foundations", "ReLU", "A", True, 1.0, "ok"))
        session.add_result(QuizResult("q2", QuestionType.MULTIPLE_CHOICE, "foundations", "Sigmoid", "B", False, 0.0, "wrong"))
        session.add_result(QuizResult("q3", QuestionType.MULTIPLE_CHOICE, "training", "SGD", "C", True, 1.0, "ok"))
        stats = session.get_stats()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["correct"], 2)
        self.assertAlmostEqual(stats["accuracy"], 2 / 3)

    def test_session_to_dict(self):
        session = QuizSession(session_id="test")
        d = session.to_dict()
        self.assertEqual(d["session_id"], "test")
        self.assertIn("stats", d)
        self.assertIn("results", d)


class TestQuizResult(unittest.TestCase):
    def test_to_dict(self):
        result = QuizResult(
            question_id="q1",
            question_type=QuestionType.MULTIPLE_CHOICE,
            section="foundations",
            topic="ReLU",
            user_answer="A",
            correct=True,
            score=1.0,
            explanation="Correct!",
            improvement_suggestion="",
        )
        d = result.to_dict()
        self.assertEqual(d["question_id"], "q1")
        self.assertEqual(d["question_type"], "multiple_choice")
        self.assertTrue(d["correct"])
        self.assertIn("timestamp", d)


class TestQuizAgent(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.agent = QuizAgent(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        self.assertIsNotNone(self.agent.question_bank)
        self.assertIsNotNone(self.agent.tracker)
        self.assertIsNotNone(self.agent.persistence)

    def test_tracker_has_topics(self):
        self.assertGreater(len(self.agent.tracker.records), 0)

    def test_start_session(self):
        session = self.agent.start_session()
        self.assertIsNotNone(session)
        self.assertIsNotNone(session.session_id)
        self.assertIs(self.agent.session, session)

    def test_start_session_with_id(self):
        session = self.agent.start_session(session_id="custom-id")
        self.assertEqual(session.session_id, "custom-id")

    def test_end_session(self):
        self.agent.start_session()
        stats = self.agent.end_session()
        self.assertIsNotNone(stats)
        self.assertIsNone(self.agent.session)

    def test_end_session_without_active(self):
        stats = self.agent.end_session()
        self.assertIsNone(stats)

    def test_get_next_question(self):
        question = self.agent.get_next_question()
        self.assertIsNotNone(question)

    def test_get_next_question_by_section(self):
        question = self.agent.get_next_question(section="foundations")
        self.assertIsNotNone(question)
        self.assertEqual(question.section, "foundations")

    def test_get_next_question_by_type(self):
        question = self.agent.get_next_question(question_type=QuestionType.CODE_COMPLETION)
        self.assertIsNotNone(question)
        self.assertEqual(question.type, QuestionType.CODE_COMPLETION)

    def test_submit_answer_mc_correct(self):
        result = self.agent.submit_answer("foundations-mc-001", "B")
        self.assertTrue(result.correct)
        self.assertEqual(result.score, 1.0)

    def test_submit_answer_mc_incorrect(self):
        result = self.agent.submit_answer("foundations-mc-001", "A")
        self.assertFalse(result.correct)
        self.assertEqual(result.score, 0.0)

    def test_submit_answer_updates_tracker(self):
        self.agent.submit_answer("foundations-mc-001", "B")
        record = self.agent.tracker.records.get("foundations::What are Neural Networks")
        self.assertIsNotNone(record)
        self.assertEqual(record.correct_count, 1)

    def test_submit_answer_records_in_session(self):
        self.agent.start_session()
        self.agent.submit_answer("foundations-mc-001", "B")
        self.assertEqual(len(self.agent.session.results), 1)

    def test_submit_answer_nonexistent_question(self):
        with self.assertRaises(ValueError):
            self.agent.submit_answer("nonexistent", "A")

    def test_submit_concept_explanation(self):
        result = self.agent.submit_answer(
            "foundations-ce-001",
            "A single perceptron can only create linear boundaries. XOR is not linearly separable, so a multi-layer network is needed with hidden layers to introduce non-linearity.",
        )
        self.assertIsInstance(result.score, float)
        self.assertGreater(result.score, 0)

    def test_save_and_load_session(self):
        self.agent.start_session(session_id="persist-test")
        self.agent.submit_answer("foundations-mc-001", "B")
        self.agent.save_session()

        agent2 = QuizAgent(storage_dir=self.test_dir)
        loaded = agent2.load_session("persist-test")
        self.assertIsNotNone(loaded)
        self.assertEqual(len(loaded.results), 1)
        self.assertTrue(loaded.results[0].correct)

    def test_get_study_recommendations(self):
        self.agent.start_session()
        self.agent.submit_answer("foundations-mc-001", "B")
        self.agent.submit_answer("foundations-mc-002", "A")
        recs = self.agent.get_study_recommendations()
        self.assertIn("weak_topics", recs)
        self.assertIn("strong_topics", recs)
        self.assertIn("overall_stats", recs)
        self.assertIn("study_path", recs)

    def test_get_explanation(self):
        explanation = self.agent.get_explanation("ReLU", "foundations")
        self.assertIn("explanation", explanation)

    def test_submit_code_completion_correct(self):
        result = self.agent.submit_answer(
            "foundations-cc-001",
            "np.dot(x, w) + bias; 1 if z > 0 else 0",
        )
        self.assertTrue(result.correct)

    def test_submit_code_completion_incorrect(self):
        result = self.agent.submit_answer(
            "foundations-cc-001",
            "wrong answer",
        )
        self.assertFalse(result.correct)

    def test_multiple_sections_coverage(self):
        sections_tested = set()
        for qid in ["training-mc-001", "challenges-mc-001", "architectures-mc-001", "evaluation-mc-001"]:
            result = self.agent.submit_answer(qid, "B")
            sections_tested.add(result.section)
        self.assertGreater(len(sections_tested), 1)

    def test_get_next_question_difficulty(self):
        question = self.agent.get_next_question(difficulty="easy", section="foundations")
        self.assertIsNotNone(question)
        self.assertEqual(question.difficulty, "easy")


if __name__ == "__main__":
    unittest.main()
