from __future__ import annotations

from pathlib import Path

from rhein.ui.history_paths import portable_data_path, resolve_history_data_path, same_data_scope


def test_saved_local_absolute_path_resolves_in_a_different_project_checkout(tmp_path: Path) -> None:
    project = tmp_path / "hosted-checkout"
    csv = project / "data" / "nasdaq" / "AAL.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("Date,Close\n2025-01-01,1\n", encoding="utf-8")
    prior_machine_path = "/Users/chloe/Documents/Rhein/data/nasdaq/AAL.csv"

    assert resolve_history_data_path(prior_machine_path, project_root=project) == csv
    assert portable_data_path(prior_machine_path, project_root=project) == "data/nasdaq/AAL.csv"


def test_data_scope_comparison_ignores_local_checkout_prefix(tmp_path: Path) -> None:
    project = tmp_path / "hosted-checkout"
    current = project / "data" / "nasdaq" / "group-a"

    assert same_data_scope(
        "/Users/chloe/Documents/Rhein/data/nasdaq/group-a",
        current,
        project_root=project,
    )
