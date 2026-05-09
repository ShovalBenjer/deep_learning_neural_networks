import json
import os
from datetime import datetime
from typing import Dict, Optional


class SessionPersistence:
    def __init__(self, storage_dir: str = ".quiz_sessions"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_session(self, session_id: str, data: Dict):
        filepath = self._session_path(session_id)
        save_data = {
            "session_id": session_id,
            "saved_at": datetime.utcnow().isoformat(),
            "data": data,
        }
        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

    def load_session(self, session_id: str) -> Optional[Dict]:
        filepath = self._session_path(session_id)
        if not os.path.exists(filepath):
            return None
        with open(filepath, "r") as f:
            wrapper = json.load(f)
        return wrapper.get("data")

    def delete_session(self, session_id: str) -> bool:
        filepath = self._session_path(session_id)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        return False

    def list_sessions(self):
        sessions = []
        for fname in os.listdir(self.storage_dir):
            if fname.endswith(".json"):
                filepath = os.path.join(self.storage_dir, fname)
                try:
                    with open(filepath, "r") as f:
                        wrapper = json.load(f)
                    sessions.append({
                        "session_id": wrapper.get("session_id"),
                        "saved_at": wrapper.get("saved_at"),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        sessions.sort(key=lambda s: s.get("saved_at", ""), reverse=True)
        return sessions

    def _session_path(self, session_id: str) -> str:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in "-_")
        return os.path.join(self.storage_dir, f"{safe_id}.json")
