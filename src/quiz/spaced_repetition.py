from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.quiz.models import TopicPerformance


class SpacedRepetitionScheduler:
    MIN_EASINESS = 1.3
    DEFAULT_EASINESS = 2.5

    def update_performance(
        self, perf: TopicPerformance, quality: int
    ) -> TopicPerformance:
        quality = max(0, min(5, quality))
        perf.total_questions += 1
        perf.last_reviewed = datetime.now(timezone.utc).isoformat()

        if quality >= 3:
            perf.correct_count += 1
            perf.streak += 1
        else:
            perf.streak = 0

        if quality >= 3:
            if perf.repetitions == 0:
                perf.interval_days = 1
            elif perf.repetitions == 1:
                perf.interval_days = 6
            else:
                perf.interval_days = int(perf.interval_days * perf.easiness_factor)
            perf.repetitions += 1
        else:
            perf.repetitions = 0
            perf.interval_days = 1

        perf.easiness_factor = max(
            self.MIN_EASINESS,
            perf.easiness_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02)),
        )

        next_date = datetime.now(timezone.utc) + timedelta(days=perf.interval_days)
        perf.next_review = next_date.isoformat()

        return perf

    def get_due_topics(
        self, performances: dict[str, TopicPerformance]
    ) -> list[str]:
        now = datetime.now(timezone.utc)
        due = []
        for topic, perf in performances.items():
            try:
                next_review = datetime.fromisoformat(perf.next_review)
                if next_review.tzinfo is None:
                    next_review = next_review.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                due.append(topic)
                continue
            if next_review <= now:
                due.append(topic)
        return due

    def get_weak_topics(
        self,
        performances: dict[str, TopicPerformance],
        threshold: float = 0.6,
        min_questions: int = 2,
    ) -> list[str]:
        weak = []
        for topic, perf in performances.items():
            if perf.total_questions >= min_questions and perf.accuracy < threshold:
                weak.append(topic)
        return sorted(weak, key=lambda t: performances[t].accuracy)

    def get_priority_topics(
        self, performances: dict[str, TopicPerformance], all_topics: list[str]
    ) -> list[str]:
        due = set(self.get_due_topics(performances))
        weak = set(self.get_weak_topics(performances))
        new = set(all_topics) - set(performances.keys())

        high = sorted(due & weak, key=lambda t: performances[t].accuracy)
        medium = sorted((due - weak) | (weak - due), key=lambda t: performances.get(t, TopicPerformance()).accuracy)
        low = sorted(new)

        return high + medium + low

    @staticmethod
    def quality_from_score(score: float) -> int:
        if score >= 0.95:
            return 5
        if score >= 0.8:
            return 4
        if score >= 0.6:
            return 3
        if score >= 0.4:
            return 2
        if score >= 0.2:
            return 1
        return 0
