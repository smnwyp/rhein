"""Portable paths for backtest artifacts viewed locally and on Streamlit."""
from __future__ import annotations

from pathlib import Path


def portable_data_path(path: str | Path, *, project_root: Path) -> str:
    """Return a repository-relative ``data/...`` path when one is available."""
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(project_root.resolve()))
    except ValueError:
        pass
    parts = candidate.parts
    if "data" in parts:
        return str(Path(*parts[parts.index("data"):]))
    return str(candidate)


def resolve_history_data_path(path: str | Path, *, project_root: Path) -> Path | None:
    """Resolve both old absolute paths and portable paths in saved history."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    portable = portable_data_path(candidate, project_root=project_root)
    resolved = project_root / portable
    return resolved if resolved.is_file() else None


def same_data_scope(left: str | Path, right: str | Path, *, project_root: Path) -> bool:
    """Compare data groups independent of the machine that produced a run."""
    return portable_data_path(left, project_root=project_root) == portable_data_path(right, project_root=project_root)
