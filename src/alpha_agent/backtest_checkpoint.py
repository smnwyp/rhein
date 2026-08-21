"""Crash-safe, resumable intermediate state for grouped backtests."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile


class JsonBacktestCheckpoint:
    def __init__(self, path: Path) -> None: self.path = path

    def load(self, key: str) -> dict | None:
        if not self.path.exists(): return None
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if value.get("key") == key else None
        except (OSError, ValueError): return None

    def save(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as temp:
            json.dump(payload, temp, ensure_ascii=False)
            temporary = Path(temp.name)
        temporary.replace(self.path)

    def clear(self, key: str) -> None:
        current = self.load(key)
        if current is not None:
            self.path.unlink(missing_ok=True)
