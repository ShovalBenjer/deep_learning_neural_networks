from datetime import datetime, timedelta, timezone

from src.quiz.models import TopicPerformance
from src.quiz.spaced_repetition import SpacedRepetitionScheduler


class TestSpacedRepetitionQuality:
    def test_quality_from_score(self):
        s = SpacedRepetitionScheduler
        assert s.quality_from_score(1.0) == 5
        assert s.quality_from_score(0.85) == 4
        assert s.quality_from_score(0.65) == 3
        assert s.quality_from_score(0.45) == 2
        assert s.quality_from_score(0.25) == 1
        assert s.quality_from_score(0.1) == 0


class TestUpdatePerformance:
    def test_correct_answer_increments(self):
        scheduler = SpacedRepetitionScheduler()
        perf = TopicPerformance(topic="test")
        updated = scheduler.update_performance(perf, quality=4)
        assert updated.total_questions == 1
        assert updated.correct_count == 1
        assert updated.streak == 1
        assert updated.repetitions == 1
        assert updated.interval_days == 1

    def test_incorrect_answer_resets(self):
        scheduler = SpacedRepetitionScheduler()
        perf = TopicPerformance(topic="test", streak=5, repetitions=3, interval_days=10)
        updated = scheduler.update_performance(perf, quality=1)
        assert updated.correct_count == 0
        assert updated.streak == 0
        assert updated.repetitions == 0
        assert updated.interval_days == 1

    def test_second_correct_sets_interval_6(self):
        scheduler = SpacedRepetitionScheduler()
        perf = TopicPerformance(topic="test", repetitions=1)
        updated = scheduler.update_performance(perf, quality=4)
        assert updated.repetitions == 2
        assert updated.interval_days == 6

    def test_easiness_factor_minimum(self):
        scheduler = SpacedRepetitionScheduler()
        perf = TopicPerformance(topic="test", easiness_factor=1.3)
        updated = scheduler.update_performance(perf, quality=0)
        assert updated.easiness_factor >= 1.3

    def test_easiness_factor_increases_on_good_quality(self):
        scheduler = SpacedRepetitionScheduler()
        perf = TopicPerformance(topic="test", easiness_factor=2.5)
        updated = scheduler.update_performance(perf, quality=5)
        assert updated.easiness_factor > 2.5


class TestDueTopics:
    def test_due_topics_past(self):
        scheduler = SpacedRepetitionScheduler()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        perfs = {"test": TopicPerformance(topic="test", next_review=past)}
        due = scheduler.get_due_topics(perfs)
        assert "test" in due

    def test_due_topics_future(self):
        scheduler = SpacedRepetitionScheduler()
        future = (datetime.now(timezone.utc) + timedelta(days=10)).isoformat()
        perfs = {"test": TopicPerformance(topic="test", next_review=future)}
        due = scheduler.get_due_topics(perfs)
        assert "test" not in due

    def test_invalid_date_considered_due(self):
        scheduler = SpacedRepetitionScheduler()
        perfs = {"test": TopicPerformance(topic="test", next_review="invalid-date")}
        due = scheduler.get_due_topics(perfs)
        assert "test" in due


class TestWeakTopics:
    def test_weak_topics_below_threshold(self):
        scheduler = SpacedRepetitionScheduler()
        perfs = {
            "weak": TopicPerformance(topic="weak", total_questions=5, correct_count=1),
            "strong": TopicPerformance(topic="strong", total_questions=5, correct_count=4),
        }
        weak = scheduler.get_weak_topics(perfs, threshold=0.6, min_questions=2)
        assert "weak" in weak
        assert "strong" not in weak

    def test_weak_topics_sorted_by_accuracy(self):
        scheduler = SpacedRepetitionScheduler()
        perfs = {
            "a": TopicPerformance(topic="a", total_questions=10, correct_count=2),
            "b": TopicPerformance(topic="b", total_questions=10, correct_count=4),
        }
        weak = scheduler.get_weak_topics(perfs, threshold=0.6, min_questions=2)
        assert weak[0] == "a"
        assert weak[1] == "b"

    def test_below_min_questions_not_weak(self):
        scheduler = SpacedRepetitionScheduler()
        perfs = {"new": TopicPerformance(topic="new", total_questions=1, correct_count=0)}
        weak = scheduler.get_weak_topics(perfs, min_questions=2)
        assert "new" not in weak


class TestPriorityTopics:
    def test_new_topics_prioritized_last(self):
        scheduler = SpacedRepetitionScheduler()
        all_topics = ["existing", "new_topic"]
        perfs = {"existing": TopicPerformance(topic="existing", total_questions=5, correct_count=4)}
        due = set(scheduler.get_due_topics(perfs))
        priority = scheduler.get_priority_topics(perfs, all_topics)
        if "existing" in due:
            assert priority.index("existing") < priority.index("new_topic")
