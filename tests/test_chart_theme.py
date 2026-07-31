from __future__ import annotations

from rhein.ui.chart_theme import event_annotation_style


def test_event_annotations_reverse_for_dark_theme() -> None:
    assert event_annotation_style("dark") == {"color": "#f8fafc", "halo_color": "#111827", "halo_width": 3}


def test_event_annotations_use_dark_text_in_light_or_unknown_theme() -> None:
    expected = {"color": "#111827", "halo_color": "#111827", "halo_width": 0}
    assert event_annotation_style("light") == expected
    assert event_annotation_style(None) == expected
