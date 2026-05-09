import unittest
from datetime import datetime, timedelta

from src.quiz.spaced_repetition import SpacedRepetitionTracker, TopicRecord


class TestTopicRecord(unittest.TestCase):
    def test_default_values(self):
        record = TopicRecord(topic="ReLU", section="foundations")
        self.assertEqual(record.topic, "ReLU")
        self.assertEqual(record.section, "foundations")
        self.assertEqual(record.ease_factor, 2.5)
        self.assertEqual(record.interval, 0)
        self.assertEqual(record.repetitions, 0)
        self.assertEqual(record.correct_count, 0)
        self.assertEqual(record.incorrect_count, 0)
        self.assertEqual(record.total_attempts, 0)

    def test_to_dict_and_from_dict(self):
        record = TopicRecord(
            topic="Sigmoid",
            section="foundations",
            ease_factor=2.3,
            interval=6,
            repetitions=3,
            correct_count=5,
            incorrect_count=2,
            total_attempts=7,
        )
        d = record.to_dict()
        restored = TopicRecord.from_dict(d)
        self.assertEqual(restored.topic, "Sigmoid")
        self.assertEqual(restored.ease_factor, 2.3)
        self.assertEqual(restored.interval, 6)
        self.assertEqual(restored.repetitions, 3)
        self.assertEqual(restored.correct_count, 5)

    def test_from_dict_ignores_unknown_keys(self):
        d = {"topic": "XOR", "section": "foundations", "unknown_key": "ignored"}
        record = TopicRecord.from_dict(d)
        self.assertEqual(record.topic, "XOR")
        self.assertFalse(hasattr(record, "unknown_key"))


class TestSpacedRepetitionTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = SpacedRepetitionTracker()

    def test_add_topic(self):
        self.tracker.add_topic("ReLU", "foundations")
        key = "foundations::ReLU"
        self.assertIn(key, self.tracker.records)
        record = self.tracker.records[key]
        self.assertEqual(record.topic, "ReLU")
        self.assertIsNotNone(record.next_review)

    def test_add_duplicate_topic(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.add_topic("ReLU", "foundations")
        self.assertEqual(len(self.tracker.records), 1)

    def test_record_correct_answer(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        key = "foundations::ReLU"
        record = self.tracker.records[key]
        self.assertEqual(record.correct_count, 1)
        self.assertEqual(record.total_attempts, 1)
        self.assertEqual(record.repetitions, 1)
        self.assertEqual(record.interval, 1)

    def test_record_incorrect_answer(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=False)
        key = "foundations::ReLU"
        record = self.tracker.records[key]
        self.assertEqual(record.incorrect_count, 1)
        self.assertEqual(record.total_attempts, 1)
        self.assertEqual(record.repetitions, 0)
        self.assertEqual(record.interval, 1)

    def test_schedule_advances_on_repeated_correct(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        interval_after_1 = self.tracker.records["foundations::ReLU"].interval
        self.assertEqual(interval_after_1, 1)

        self.tracker.record_result("ReLU", "foundations", correct=True)
        interval_after_2 = self.tracker.records["foundations::ReLU"].interval
        self.assertEqual(interval_after_2, 6)

        self.tracker.record_result("ReLU", "foundations", correct=True)
        interval_after_3 = self.tracker.records["foundations::ReLU"].interval
        self.assertGreater(interval_after_3, 6)

    def test_resets_on_wrong_answer(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        self.tracker.record_result("ReLU", "foundations", correct=True)
        self.assertEqual(self.tracker.records["foundations::ReLU"].repetitions, 2)

        self.tracker.record_result("ReLU", "foundations", correct=False)
        record = self.tracker.records["foundations::ReLU"]
        self.assertEqual(record.repetitions, 0)
        self.assertEqual(record.interval, 1)

    def test_get_due_topics(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.add_topic("Sigmoid", "foundations")
        due = self.tracker.get_due_topics()
        self.assertEqual(len(due), 2)

    def test_get_due_topics_filtered_by_section(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.add_topic("SGD", "training")
        due = self.tracker.get_due_topics(section="foundations")
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0].section, "foundations")

    def test_get_weak_topics(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.add_topic("Sigmoid", "foundations")
        for _ in range(3):
            self.tracker.record_result("ReLU", "foundations", correct=False)
            self.tracker.record_result("Sigmoid", "foundations", correct=True)
        weak = self.tracker.get_weak_topics(threshold=0.5)
        weak_names = [r.topic for r in weak]
        self.assertIn("ReLU", weak_names)
        self.assertNotIn("Sigmoid", weak_names)

    def test_get_strong_topics(self):
        self.tracker.add_topic("ReLU", "foundations")
        for _ in range(4):
            self.tracker.record_result("ReLU", "foundations", correct=True)
        strong = self.tracker.get_strong_topics(threshold=0.8, min_attempts=3)
        self.assertEqual(len(strong), 1)
        self.assertEqual(strong[0].topic, "ReLU")

    def test_get_section_stats(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.add_topic("Sigmoid", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        self.tracker.record_result("Sigmoid", "foundations", correct=False)
        stats = self.tracker.get_section_stats("foundations")
        self.assertEqual(stats["topics"], 2)
        self.assertEqual(stats["total_attempts"], 2)
        self.assertAlmostEqual(stats["accuracy"], 0.5)

    def test_get_section_stats_empty(self):
        stats = self.tracker.get_section_stats("nonexistent")
        self.assertEqual(stats["topics"], 0)
        self.assertEqual(stats["accuracy"], 0.0)

    def test_get_overall_stats(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        stats = self.tracker.get_overall_stats()
        self.assertEqual(stats["total_topics"], 1)
        self.assertEqual(stats["total_attempts"], 1)
        self.assertIn("foundations", stats["sections"])

    def test_get_overall_stats_empty(self):
        stats = self.tracker.get_overall_stats()
        self.assertEqual(stats["total_topics"], 0)

    def test_ease_factor_minimum(self):
        self.tracker.add_topic("ReLU", "foundations")
        for _ in range(10):
            self.tracker.record_result("ReLU", "foundations", correct=False)
        record = self.tracker.records["foundations::ReLU"]
        self.assertGreaterEqual(record.ease_factor, SpacedRepetitionTracker.MIN_EASE_FACTOR)

    def test_serialization_round_trip(self):
        self.tracker.add_topic("ReLU", "foundations")
        self.tracker.record_result("ReLU", "foundations", correct=True)
        self.tracker.add_topic("SGD", "training")
        self.tracker.record_result("SGD", "training", correct=False)

        data = self.tracker.to_dict()
        restored = SpacedRepetitionTracker.from_dict(data)

        self.assertEqual(len(restored.records), 2)
        key = "foundations::ReLU"
        self.assertEqual(restored.records[key].correct_count, 1)
        self.assertEqual(restored.records[key].topic, "ReLU")

    def test_auto_add_on_record_result(self):
        self.tracker.record_result("NewTopic", "foundations", correct=True)
        key = "foundations::NewTopic"
        self.assertIn(key, self.tracker.records)
        self.assertEqual(self.tracker.records[key].correct_count, 1)


if __name__ == "__main__":
    unittest.main()
