import unittest

from src.quiz.question_bank import QuestionBank, Question, QuestionType, SECTIONS


class TestQuestionBank(unittest.TestCase):
    def setUp(self):
        self.bank = QuestionBank()

    def test_has_questions(self):
        self.assertGreater(len(self.bank.questions), 0)

    def test_get_question_by_id(self):
        question = self.bank.get_question("foundations-mc-001")
        self.assertIsNotNone(question)
        self.assertEqual(question.id, "foundations-mc-001")
        self.assertEqual(question.type, QuestionType.MULTIPLE_CHOICE)

    def test_get_question_nonexistent(self):
        question = self.bank.get_question("nonexistent")
        self.assertIsNone(question)

    def test_get_questions_by_section(self):
        questions = self.bank.get_questions_by_section("foundations")
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertEqual(q.section, "foundations")

    def test_get_questions_by_topic(self):
        questions = self.bank.get_questions_by_topic("Perceptron Model", "foundations")
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertEqual(q.topic, "Perceptron Model")

    def test_get_questions_by_type_multiple_choice(self):
        questions = self.bank.get_questions_by_type(QuestionType.MULTIPLE_CHOICE)
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertEqual(q.type, QuestionType.MULTIPLE_CHOICE)

    def test_get_questions_by_type_code_completion(self):
        questions = self.bank.get_questions_by_type(QuestionType.CODE_COMPLETION)
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertEqual(q.type, QuestionType.CODE_COMPLETION)

    def test_get_questions_by_type_concept_explanation(self):
        questions = self.bank.get_questions_by_type(QuestionType.CONCEPT_EXPLANATION)
        self.assertGreater(len(questions), 0)
        for q in questions:
            self.assertEqual(q.type, QuestionType.CONCEPT_EXPLANATION)

    def test_get_questions_by_difficulty(self):
        for difficulty in ["easy", "medium", "hard"]:
            questions = self.bank.get_questions_by_difficulty(difficulty)
            self.assertGreater(len(questions), 0, f"No questions for difficulty: {difficulty}")
            for q in questions:
                self.assertEqual(q.difficulty, difficulty)

    def test_get_random_question(self):
        question = self.bank.get_random_question()
        self.assertIsNotNone(question)
        self.assertIsInstance(question, Question)

    def test_get_random_question_with_filters(self):
        question = self.bank.get_random_question(
            section="foundations",
            qtype=QuestionType.MULTIPLE_CHOICE,
            difficulty="easy",
        )
        self.assertIsNotNone(question)
        self.assertEqual(question.section, "foundations")
        self.assertEqual(question.type, QuestionType.MULTIPLE_CHOICE)
        self.assertEqual(question.difficulty, "easy")

    def test_get_random_question_no_match(self):
        question = self.bank.get_random_question(section="nonexistent_section")
        self.assertIsNone(question)

    def test_get_all_sections(self):
        sections = self.bank.get_all_sections()
        self.assertIn("foundations", sections)
        self.assertIn("training", sections)
        self.assertIn("challenges", sections)
        self.assertIn("architectures", sections)
        self.assertIn("evaluation", sections)

    def test_get_section_topics(self):
        topics = self.bank.get_section_topics("foundations")
        self.assertGreater(len(topics), 0)
        self.assertIn("Perceptron Model", topics)

    def test_get_section_topics_nonexistent(self):
        topics = self.bank.get_section_topics("nonexistent")
        self.assertEqual(topics, [])

    def test_add_question(self):
        new_q = Question(
            id="test-custom-001",
            type=QuestionType.MULTIPLE_CHOICE,
            section="foundations",
            topic="Custom Topic",
            difficulty="easy",
            question_text="Test question?",
            options={"A": "Yes", "B": "No"},
            correct_answer="A",
            explanation="Test explanation",
        )
        initial_count = len(self.bank.questions)
        self.bank.add_question(new_q)
        self.assertEqual(len(self.bank.questions), initial_count + 1)
        retrieved = self.bank.get_question("test-custom-001")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.question_text, "Test question?")

    def test_question_fields(self):
        mc_question = self.bank.get_question("foundations-mc-001")
        self.assertIsNotNone(mc_question.options)
        self.assertIsNotNone(mc_question.correct_answer)
        self.assertIsNotNone(mc_question.explanation)

        cc_question = self.bank.get_questions_by_type(QuestionType.CODE_COMPLETION)[0]
        self.assertIsNotNone(cc_question.code_template)
        self.assertIsNotNone(cc_question.blank_description)

        ce_question = self.bank.get_questions_by_type(QuestionType.CONCEPT_EXPLANATION)[0]
        self.assertIsNotNone(ce_question.key_concepts)
        self.assertIsNotNone(ce_question.grading_rubric)
        self.assertIsNotNone(ce_question.sample_answer)

    def test_all_sections_have_questions(self):
        for section_key in SECTIONS:
            questions = self.bank.get_questions_by_section(section_key)
            self.assertGreater(
                len(questions), 0,
                f"Section {section_key} has no questions",
            )

    def test_all_questions_have_valid_type(self):
        for qid, q in self.bank.questions.items():
            self.assertIsInstance(q.type, QuestionType)


if __name__ == "__main__":
    unittest.main()
