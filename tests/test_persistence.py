import json
import tempfile
from pathlib import Path

from src.quiz.models import QuizSession, TopicPerformance
from src.quiz.persistence import PersistenceManager


class TestPersistenceManager:
    def test_save_and_load_session(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        session = QuizSession(
            questions=[],
            answers=[],
        )
        path = pm.save_session(session)
        assert path.exists()

        loaded = pm.load_session(session.id)
        assert loaded is not None
        assert loaded.id == session.id

    def test_load_nonexistent_session(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        result = pm.load_session("nonexistent")
        assert result is None

    def test_list_sessions(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        s1 = QuizSession()
        s2 = QuizSession()
        pm.save_session(s1)
        pm.save_session(s2)
        sessions = pm.list_sessions()
        assert len(sessions) == 2

    def test_save_and_load_performance(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        perfs = {
            "test": TopicPerformance(
                topic="test", total_questions=10, correct_count=7
            ),
        }
        pm.save_performance(perfs)
        loaded = pm.load_performance()
        assert "test" in loaded
        assert loaded["test"].total_questions == 10
        assert loaded["test"].correct_count == 7

    def test_load_empty_performance(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        loaded = pm.load_performance()
        assert loaded == {}

    def test_delete_session(self, tmp_path):
        pm = PersistenceManager(data_dir=tmp_path)
        session = QuizSession()
        pm.save_session(session)
        assert pm.delete_session(session.id) is True
        assert pm.load_session(session.id) is None
        assert pm.delete_session(session.id) is False
