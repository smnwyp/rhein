"""Private, file-backed cache for fully validated interpreter results.

This cache is intentionally an optimisation only: a cached payload is parsed and
validated again by :class:`StrategyInterpreterService` before it is returned.
It is stored below ``.cache/`` because source strategy prose may be sensitive
and must not become part of the public saved-strategy product data.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from alpha_agent.domain.interpretation import StrategyInterpretationRequest


_CACHE_FORMAT_VERSION = "parsed-interpretation-v1"


def interpretation_cache_key(request: StrategyInterpretationRequest) -> str:
    """Hash every interpretation-relevant request value, deterministically."""
    payload = {
        "format": _CACHE_FORMAT_VERSION,
        "request": request.model_dump(mode="json", exclude={"repair_instruction"}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


class ParsedInterpretationCache(Protocol):
    def get(self, request: StrategyInterpretationRequest) -> dict[str, Any] | None: ...
    def put(self, request: StrategyInterpretationRequest, result: dict[str, Any]) -> None: ...
    def delete(self, request: StrategyInterpretationRequest) -> None: ...


class JsonParsedInterpretationCache:
    """Small atomic JSON cache, deliberately without a public-data dependency."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def get(self, request: StrategyInterpretationRequest) -> dict[str, Any] | None:
        entries = self._read()
        value = entries.get(interpretation_cache_key(request))
        return value if isinstance(value, dict) else None

    def put(self, request: StrategyInterpretationRequest, result: dict[str, Any]) -> None:
        entries = self._read()
        entries[interpretation_cache_key(request)] = result
        self._write(entries)

    def delete(self, request: StrategyInterpretationRequest) -> None:
        entries = self._read()
        if entries.pop(interpretation_cache_key(request), None) is not None:
            self._write(entries)

    def _read(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            value = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A cache must never make the interpreter unavailable. A corrupt
            # cache is treated as empty and overwritten by the next success.
            return {}
        return value if isinstance(value, dict) else {}

    def _write(self, entries: dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with NamedTemporaryFile("w", encoding="utf-8", dir=self._path.parent, delete=False) as temporary:
            temporary.write(serialized)
            temporary_path = Path(temporary.name)
        temporary_path.replace(self._path)
