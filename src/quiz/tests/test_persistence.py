import unittest
import os
import json
import tempfile
import shutil

from src.quiz.persistence import SessionPersistence


class TestSessionPersistence(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.persistence = SessionPersistence(storage_dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_and_load_session(self):
        data = {"key": "value", "count": 42}
        self.persistence.save_session("test-session-1", data)
        loaded = self.persistence.load_session("test-session-1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["key"], "value")
        self.assertEqual(loaded["count"], 42)

    def test_load_nonexistent_session(self):
        loaded = self.persistence.load_session("nonexistent")
        self.assertIsNone(loaded)

    def test_delete_session(self):
        self.persistence.save_session("test-delete", {"data": True})
        result = self.persistence.delete_session("test-delete")
        self.assertTrue(result)
        loaded = self.persistence.load_session("test-delete")
        self.assertIsNone(loaded)

    def test_delete_nonexistent_session(self):
        result = self.persistence.delete_session("nonexistent")
        self.assertFalse(result)

    def test_list_sessions(self):
        self.persistence.save_session("session-a", {"a": 1})
        self.persistence.save_session("session-b", {"b": 2})
        sessions = self.persistence.list_sessions()
        self.assertEqual(len(sessions), 2)
        ids = [s["session_id"] for s in sessions]
        self.assertIn("session-a", ids)
        self.assertIn("session-b", ids)

    def test_save_overwrites(self):
        self.persistence.save_session("overwrite-test", {"version": 1})
        self.persistence.save_session("overwrite-test", {"version": 2})
        loaded = self.persistence.load_session("overwrite-test")
        self.assertEqual(loaded["version"], 2)

    def test_session_path_sanitize(self):
        self.persistence.save_session("test/path\\here", {"safe": True})
        loaded = self.persistence.load_session("test/path\\here")
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded["safe"])

    def test_saved_data_includes_metadata(self):
        self.persistence.save_session("meta-test", {"data": True})
        filepath = os.path.join(self.test_dir, "meta-test.json")
        with open(filepath) as f:
            wrapper = json.load(f)
        self.assertIn("session_id", wrapper)
        self.assertIn("saved_at", wrapper)
        self.assertIn("data", wrapper)
        self.assertEqual(wrapper["data"]["data"], True)

    def test_complex_data(self):
        data = {
            "results": [
                {"question_id": "q1", "correct": True, "score": 1.0},
                {"question_id": "q2", "correct": False, "score": 0.3},
            ],
            "tracker": {"key1": {"nested": True}},
        }
        self.persistence.save_session("complex-test", data)
        loaded = self.persistence.load_session("complex-test")
        self.assertEqual(len(loaded["results"]), 2)
        self.assertTrue(loaded["tracker"]["key1"]["nested"])


if __name__ == "__main__":
    unittest.main()
