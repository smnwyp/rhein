from __future__ import annotations

from rhein.paths import GROUP_ROOT
from rhein.strategy import active_condition_ids, parse_ints, parse_percent_ranges


def test_strategy_parsers_and_condition_ids() -> None:
    assert parse_ints("3, 4") == (3, 4)
    assert parse_percent_ranges("2-2.5") == [(0.02, 0.025)]
    assert active_condition_ids({"use_signal_band": True, "use_forced_exit": False}) == [
        "T0-01", "T0-02", "T0-03", "T0-04", "T0-05", "T0-06", "T0-07", "T0-08", "EN-01",
        "EX-01", "EX-02", "EX-03", "EX-05",
    ]


def test_nine_mature_group_directories_exist() -> None:
    groups = [path for path in GROUP_ROOT.glob("[0-9][0-9]_*") if path.is_dir() and not path.name.startswith("10_")]
    assert len(groups) == 9


def test_streamlit_entrypoint_renders() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py")
    app.run(timeout=45)

    assert not app.exception
    assert len(app.tabs) == 5
    scope = next(widget for widget in app.selectbox if widget.label == "数据范围")
    assert len([item for item in scope.options if "流动性" in str(item)]) == 9
