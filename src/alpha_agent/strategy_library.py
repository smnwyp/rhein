"""Small file-backed, cross-group library for parsed strategy definitions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID, uuid4

from pydantic import Field

from alpha_agent.domain.interpretation import InterpretationNote
from alpha_agent.domain.indicators import DSLModel
from alpha_agent.domain.strategy import AnyStrategyDefinition
from alpha_agent.errors import AlphaAgentError


class SavedStrategy(DSLModel):
    strategy_id: UUID = Field(default_factory=uuid4)
    strategy_name: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    original_language: str = Field(min_length=1)
    strategy: AnyStrategyDefinition | None = None
    assumptions: list[InterpretationNote] = Field(default_factory=list)
    warnings: list[InterpretationNote] = Field(default_factory=list)


class StrategyLibraryError(AlphaAgentError):
    code = "strategy_library_error"


class JsonStrategyLibrary:
    """Global repository-local library. It deliberately has no group foreign key."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def list(self) -> list[SavedStrategy]:
        if not self._path.exists():
            return []
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            return [SavedStrategy.model_validate_json(json.dumps(item)) for item in payload]
        except (OSError, ValueError) as error:
            raise StrategyLibraryError("saved strategy library cannot be read", details={"path": str(self._path)}) from error

    def save(self, strategy: SavedStrategy) -> SavedStrategy:
        entries = self.list()
        entries.append(strategy)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps([entry.model_dump(mode="json") for entry in entries], ensure_ascii=False, indent=2)
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=self._path.parent, delete=False) as temporary:
                temporary.write(serialized)
                temporary_path = Path(temporary.name)
            temporary_path.replace(self._path)
        except OSError as error:
            raise StrategyLibraryError("saved strategy library cannot be written", details={"path": str(self._path)}) from error
        return strategy
