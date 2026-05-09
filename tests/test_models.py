from src.quiz.models import (
    QuizAnswer,
    QuizOption,
    QuizQuestion,
    QuizSession,
    QuizType,
    TopicPerformance,
)


class TestQuizOption:
    def test_creation(self):
        opt = QuizOption(label="A", text="Option A", is_correct=True)
        assert opt.label == "A"
        assert opt.text == "Option A"
        assert opt.is_correct is True

    def test_field_access(self):
        opt = QuizOption(label="B", text="Option B", is_correct=False)
        assert opt.label == "B"
        assert opt.text == "Option B"
        assert opt.is_correct is False


class TestQuizQuestion:
    def test_creation_defaults(self):
        q = QuizQuestion()
        assert q.quiz_type == QuizType.MULTIPLE_CHOICE
        assert q.difficulty == 3
        assert q.options == []

    def test_to_dict_and_from_dict(self):
        q = QuizQuestion(
            quiz_type=QuizType.MULTIPLE_CHOICE,
            topic="neural_networks",
            section="Intro",
            question_text="What is a neuron?",
            options=[
                QuizOption("A", "A cell", True),
                QuizOption("B", "A rock", False),
            ],
            correct_answer="A",
            explanation="Neurons are cells.",
            difficulty=2,
        )
        d = q.to_dict()
        restored = QuizQuestion.from_dict(d)
        assert restored.quiz_type == QuizType.MULTIPLE_CHOICE
        assert restored.topic == "neural_networks"
        assert len(restored.options) == 2
        assert restored.options[0].is_correct is True
        assert restored.correct_answer == "A"

    def test_code_completion_question(self):
        q = QuizQuestion(
            quiz_type=QuizType.CODE_COMPLETION,
            topic="pytorch",
            question_text="Fill in the blank:",
            code_template="import torch\nx = torch.__BLANK__([1, 2, 3])",
            correct_answer="tensor",
        )
        d = q.to_dict()
        restored = QuizQuestion.from_dict(d)
        assert restored.quiz_type == QuizType.CODE_COMPLETION
        assert "__BLANK__" in restored.code_template


class TestQuizAnswer:
    def test_to_dict_and_from_dict(self):
        a = QuizAnswer(
            question_id="q1",
            user_answer="A",
            is_correct=True,
            feedback="Correct!",
            score=1.0,
        )
        d = a.to_dict()
        restored = QuizAnswer.from_dict(d)
        assert restored.question_id == "q1"
        assert restored.is_correct is True
        assert restored.score == 1.0


class TestTopicPerformance:
    def test_accuracy_zero_questions(self):
        perf = TopicPerformance(topic="test")
        assert perf.accuracy == 0.0

    def test_accuracy_calculation(self):
        perf = TopicPerformance(topic="test", total_questions=10, correct_count=7)
        assert abs(perf.accuracy - 0.7) < 0.001

    def test_to_dict_and_from_dict(self):
        perf = TopicPerformance(
            topic="backprop",
            total_questions=5,
            correct_count=3,
            streak=2,
            easiness_factor=2.3,
            interval_days=6,
            repetitions=3,
        )
        d = perf.to_dict()
        restored = TopicPerformance.from_dict(d)
        assert restored.topic == "backprop"
        assert restored.total_questions == 5
        assert restored.easiness_factor == 2.3


class TestQuizSession:
    def test_empty_session_score(self):
        session = QuizSession()
        assert session.score == 0.0

    def test_session_score(self):
        session = QuizSession(
            answers=[
                QuizAnswer(score=1.0),
                QuizAnswer(score=0.5),
                QuizAnswer(score=0.0),
            ]
        )
        expected = (1.0 + 0.5 + 0.0) / 3
        assert abs(session.score - expected) < 0.001

    def test_to_dict_and_from_dict(self):
        session = QuizSession(
            questions=[
                QuizQuestion(topic="test", question_text="Q1"),
            ],
            answers=[
                QuizAnswer(question_id="q1", score=1.0, is_correct=True),
            ],
            topic_performances={
                "test": TopicPerformance(topic="test", total_questions=1, correct_count=1),
            },
        )
        d = session.to_dict()
        restored = QuizSession.from_dict(d)
        assert len(restored.questions) == 1
        assert len(restored.answers) == 1
        assert "test" in restored.topic_performances
        assert restored.topic_performances["test"].total_questions == 1
