"""Tests for the Quiz Agent with Spaced Repetition"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import unittest
import tempfile
from datetime import datetime, timedelta

from quiz_agent import QuizAgent, SM2Scheduler, Question, UserProgress


class TestSM2Scheduler(unittest.TestCase):
    def setUp(self):
        self.scheduler = SM2Scheduler()
    
    def test_initial_interval(self):
        self.assertEqual(self.scheduler.calculate_interval(0, 2.5), 1)
    
    def test_second_interval(self):
        self.assertEqual(self.scheduler.calculate_interval(1, 2.5), 6)
    
    def test_later_intervals(self):
        self.assertEqual(self.scheduler.calculate_interval(2, 2.5), 5)
        self.assertEqual(self.scheduler.calculate_interval(3, 2.5), 8)
    
    def test_ease_factor_increases(self):
        new_ef = self.scheduler.calculate_ease_factor(2.5, 5)
        self.assertGreater(new_ef, 2.5)
    
    def test_ease_factor_decreases(self):
        new_ef = self.scheduler.calculate_ease_factor(2.5, 2)
        self.assertLess(new_ef, 2.5)
    
    def test_ease_factor_minimum(self):
        new_ef = self.scheduler.calculate_ease_factor(1.35, 2)
        self.assertGreaterEqual(new_ef, 1.3)
    
    def test_quality_from_correct(self):
        self.assertEqual(self.scheduler.quality_from_correct(True), 4)
        self.assertEqual(self.scheduler.quality_from_correct(False), 2)
        self.assertEqual(self.scheduler.quality_from_correct(True, 3000), 5)


class TestQuizAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QuizAgent("test_user")
        self.agent._save_progress()
    
    def tearDown(self):
        progress_file = f"progress_test_user.json"
        if os.path.exists(progress_file):
            os.remove(progress_file)
    
    def test_questions_loaded(self):
        self.assertGreater(len(self.agent.questions), 0)
    
    def test_get_due_questions(self):
        due = self.agent.get_due_questions()
        self.assertGreater(len(due), 0)
    
    def test_submit_correct_answer(self):
        qid = list(self.agent.questions.keys())[0]
        question = self.agent.questions[qid]
        result = self.agent.submit_answer(qid, question.correct_answer)
        
        self.assertTrue(result["correct"])
        self.assertIn("explanation", result)
        self.assertIn("next_review", result)
    
    def test_submit_invalid_question_id(self):
        with self.assertRaises(ValueError):
            self.agent.submit_answer("nonexistent_id", 0)

    def test_spaced_repetition_interval(self):
        qid = list(self.agent.questions.keys())[0]
        
        initial_progress = self.agent.user_progress.get(qid)
        if initial_progress is None:
            initial_progress = UserProgress(user_id="test_user", question_id=qid)
        
        initial_interval = initial_progress.interval_days
        
        for _ in range(3):
            self.agent.submit_answer(qid, 0)
        
        progress = self.agent.user_progress[qid]
        self.assertGreater(progress.interval_days, initial_interval)
    
    def test_incorrect_answer_resets_repetitions(self):
        qid = list(self.agent.questions.keys())[0]
        
        self.agent.submit_answer(qid, 0)
        self.agent.submit_answer(qid, 0)
        
        question = self.agent.questions[qid]
        wrong_answer = 1 if question.correct_answer != 1 else 0
        self.agent.submit_answer(qid, wrong_answer)
        
        progress = self.agent.user_progress[qid]
        self.assertEqual(progress.repetitions, 0)
    
    def test_get_stats(self):
        stats = self.agent.get_stats()
        
        self.assertIn("total_questions", stats)
        self.assertIn("answered_questions", stats)
        self.assertIn("mastered_questions", stats)
        self.assertIn("completion_rate", stats)
    
    def test_export_anki(self):
        filename = self.agent.export_anki()
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)
    
    def test_progress_persistence(self):
        qid = list(self.agent.questions.keys())[0]
        self.agent.submit_answer(qid, 0)
        
        progress = self.agent.user_progress[qid]
        
        new_agent = QuizAgent("test_user")
        self.assertIn(qid, new_agent.user_progress)
        self.assertEqual(new_agent.user_progress[qid].repetitions, progress.repetitions)


class TestQuestionBank(unittest.TestCase):
    def test_questions_have_required_fields(self):
        agent = QuizAgent("validation")
        
        for qid, question in agent.questions.items():
            self.assertIsNotNone(question.id)
            self.assertIsNotNone(question.section)
            self.assertIsNotNone(question.question)
            self.assertIsNotNone(question.choices)
            self.assertGreaterEqual(len(question.choices), 2)
            self.assertIsNotNone(question.correct_answer)
            self.assertIsNotNone(question.explanation)

    def test_questions_cover_all_sections(self):
        agent = QuizAgent("coverage")
        sections = set(q.section for q in agent.questions.values())
        
        expected_sections = {"Foundations", "Activation Functions", "Training", 
                           "Regularization", "CNN", "RNN", "Transformers", "Evaluation"}
        
        for section in expected_sections:
            self.assertIn(section, sections)


if __name__ == "__main__":
    unittest.main()