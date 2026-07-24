from __future__ import annotations

import json

import pytest

from rhein.ui.persistence import load_saved_combos, normalize_parameters, save_combo


def test_normalize_parameters_migrates_legacy_baseline_flags() -> None:
    params = normalize_parameters({"use_entry_close_above_fast_sma": False, "hard_stop_days": [3, 4]})
    assert params["use_baseline_close_above_fast_sma"] is False
    assert "use_entry_close_above_fast_sma" not in params
    assert params["hard_stop_days"] == (3, 4)
    assert params["forced_exit_day"] == 5


def test_save_and_load_combo_are_scoped_to_one_group(tmp_path) -> None:
    group = tmp_path / "03_group"
    group.mkdir()
    (group / "group_manifest.csv").write_text("symbol\nNVDA\n", encoding="utf-8")
    params = normalize_parameters({"hard_stop_days": (3, 4)})
    save_combo(str(group), "我的组合", params)
    combos = load_saved_combos(str(group))
    assert len(combos) == 1
    assert combos[0]["name"] == "我的组合"
    assert combos[0]["parameters"]["hard_stop_days"] == [3, 4]
    with pytest.raises(ValueError, match="同名"):
        save_combo(str(group), "我的组合", params)
    payload = json.loads((group / "saved_combos.json").read_text(encoding="utf-8"))
    assert payload["group_folder"] == "03_group"
