import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class TopicRecord:
    topic: str
    section: str
    ease_factor: float = 2.5
    interval: int = 0
    repetitions: int = 0
    last_reviewed: Optional[str] = None
    next_review: Optional[str] = None
    correct_count: int = 0
    incorrect_count: int = 0
    total_attempts: int = 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SpacedRepetitionTracker:
    MIN_EASE_FACTOR = 1.3
    DEFAULT_EASE_FACTOR = 2.5

    GRADE_NEW = 0
    GRADE_HARD = 1
    GRADE_GOOD = 2
    GRADE_EASY = 3

    def __init__(self):
        self.records: Dict[str, TopicRecord] = {}

    def add_topic(self, topic: str, section: str):
        key = self._make_key(topic, section)
        if key not in self.records:
            self.records[key] = TopicRecord(topic=topic, section=section)
            self._schedule_new(key)

    def record_result(self, topic: str, section: str, correct: bool):
        key = self._make_key(topic, section)
        if key not in self.records:
            self.add_topic(topic, section)

        record = self.records[key]
        record.total_attempts += 1

        if correct:
            record.correct_count += 1
            grade = self.GRADE_GOOD
        else:
            record.incorrect_count += 1
            grade = self.GRADE_HARD if record.repetitions > 0 else self.GRADE_NEW

        self._update_schedule(key, grade)

    def get_due_topics(self, section: Optional[str] = None) -> List[TopicRecord]:
        now = datetime.utcnow()
        due = []
        for record in self.records.values():
            if record.next_review is None:
                due.append(record)
                continue
            next_dt = datetime.fromisoformat(record.next_review)
            if next_dt <= now:
                if section is None or record.section == section:
                    due.append(record)
        due.sort(key=lambda r: r.next_review or datetime.min.isoformat())
        return due

    def get_weak_topics(self, threshold: float = 0.5, min_attempts: int = 2) -> List[TopicRecord]:
        weak = []
        for record in self.records.values():
            if record.total_attempts >= min_attempts:
                accuracy = record.correct_count / record.total_attempts
                if accuracy < threshold:
                    weak.append(record)
        weak.sort(key=lambda r: r.correct_count / max(r.total_attempts, 1))
        return weak

    def get_strong_topics(self, threshold: float = 0.8, min_attempts: int = 3) -> List[TopicRecord]:
        strong = []
        for record in self.records.values():
            if record.total_attempts >= min_attempts:
                accuracy = record.correct_count / record.total_attempts
                if accuracy >= threshold:
                    strong.append(record)
        strong.sort(key=lambda r: r.correct_count / max(r.total_attempts, 1), reverse=True)
        return strong

    def get_section_stats(self, section: str) -> Dict:
        section_records = [r for r in self.records.values() if r.section == section]
        if not section_records:
            return {"section": section, "topics": 0, "accuracy": 0.0, "total_attempts": 0}
        total_correct = sum(r.correct_count for r in section_records)
        total_attempts = sum(r.total_attempts for r in section_records)
        return {
            "section": section,
            "topics": len(section_records),
            "accuracy": total_correct / max(total_attempts, 1),
            "total_attempts": total_attempts,
        }

    def get_overall_stats(self) -> Dict:
        if not self.records:
            return {"total_topics": 0, "total_attempts": 0, "overall_accuracy": 0.0, "sections": {}}
        sections = {}
        for record in self.records.values():
            if record.section not in sections:
                sections[record.section] = {"correct": 0, "attempts": 0, "topics": 0}
            sections[record.section]["correct"] += record.correct_count
            sections[record.section]["attempts"] += record.total_attempts
            sections[record.section]["topics"] += 1

        total_correct = sum(s["correct"] for s in sections.values())
        total_attempts = sum(s["attempts"] for s in sections.values())

        return {
            "total_topics": len(self.records),
            "total_attempts": total_attempts,
            "overall_accuracy": total_correct / max(total_attempts, 1),
            "sections": {
                s: {"accuracy": d["correct"] / max(d["attempts"], 1), "topics": d["topics"]}
                for s, d in sections.items()
            },
        }

    def _make_key(self, topic: str, section: str) -> str:
        return f"{section}::{topic}"

    def _schedule_new(self, key: str):
        record = self.records[key]
        now = datetime.utcnow()
        record.next_review = now.isoformat()
        record.last_reviewed = now.isoformat()

    def _update_schedule(self, key: str, grade: int):
        record = self.records[key]
        now = datetime.utcnow()
        record.last_reviewed = now.isoformat()

        if grade < 2:
            record.repetitions = 0
            record.interval = 1
        else:
            if record.repetitions == 0:
                record.interval = 1
            elif record.repetitions == 1:
                record.interval = 6
            else:
                record.interval = math.ceil(record.interval * record.ease_factor)
            record.repetitions += 1

        record.ease_factor = max(
            self.MIN_EASE_FACTOR,
            record.ease_factor + (0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)),
        )

        next_dt = now + timedelta(days=record.interval)
        record.next_review = next_dt.isoformat()

    def to_dict(self) -> Dict:
        return {key: record.to_dict() for key, record in self.records.items()}

    @classmethod
    def from_dict(cls, data: Dict) -> "SpacedRepetitionTracker":
        tracker = cls()
        for key, record_data in data.items():
            tracker.records[key] = TopicRecord.from_dict(record_data)
        return tracker
