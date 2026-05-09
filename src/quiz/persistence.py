from __future__ import annotations

import json
import os
from pathlib import Path

from src.quiz.models import QuizSession, TopicPerformance


DEFAULT_DATA_DIR = Path.home() / ".quiz_agent"


class PersistenceManager:
    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self.sessions_dir = self.data_dir / "sessions"
        self.performance_file = self.data_dir / "performance.json"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: QuizSession) -> Path:
        filepath = self.sessions_dir / f"{session.id}.json"
        with open(filepath, "w") as f:
            json.dump(session.to_dict(), f, indent=2)
        return filepath

    def load_session(self, session_id: str) -> QuizSession | None:
        filepath = self.sessions_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath) as f:
            data = json.load(f)
        return QuizSession.from_dict(data)

    def list_sessions(self) -> list[dict]:
        sessions = []
        for filepath in self.sessions_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                sessions.append({
                    "id": data.get("id", ""),
                    "created_at": data.get("created_at", ""),
                    "is_complete": data.get("is_complete", False),
                    "num_questions": len(data.get("questions", [])),
                    "num_answers": len(data.get("answers", [])),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(sessions, key=lambda s: s.get("created_at", ""), reverse=True)

    def save_performance(self, performances: dict[str, TopicPerformance]) -> None:
        data = {k: v.to_dict() for k, v in performances.items()}
        with open(self.performance_file, "w") as f:
            json.dump(data, f, indent=2)

    def load_performance(self) -> dict[str, TopicPerformance]:
        if not self.performance_file.exists():
            return {}
        with open(self.performance_file) as f:
            data = json.load(f)
        return {k: TopicPerformance.from_dict(v) for k, v in data.items()}

    def delete_session(self, session_id: str) -> bool:
        filepath = self.sessions_dir / f"{session_id}.json"
        if filepath.exists():
            os.remove(filepath)
            return True
        return False
