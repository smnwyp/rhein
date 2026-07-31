"""Theme-aware presentation values for interactive charts."""
from __future__ import annotations

from typing import Literal, TypedDict


ThemeType = Literal["dark", "light"] | None


class EventAnnotationStyle(TypedDict):
    """Colours used for the trade-event leader, point, and label halo."""

    color: str
    halo_color: str
    halo_width: int


def event_annotation_style(theme_type: ThemeType) -> EventAnnotationStyle:
    """Return high-contrast event-label colours for the active Streamlit theme.

    ``st.context.theme.type`` can briefly be ``None`` during the first render
    or a theme transition.  Treat that state as light: black text with a white
    halo remains legible on Streamlit's default canvas and avoids white text
    disappearing before the context is available.
    """
    if theme_type == "dark":
        return {"color": "#f8fafc", "halo_color": "#111827", "halo_width": 3}
    # A thick white stroke is visually interpreted as white lettering on the
    # pale Vega canvas, because it can cover a small bold glyph completely.
    # Light mode therefore uses a solid dark label with no halo.
    return {"color": "#111827", "halo_color": "#111827", "halo_width": 0}
