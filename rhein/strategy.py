"""Compatibility facade for strategy-domain helpers.

New code should import from :mod:`rhein.domain.conditions`.  This module keeps
the long-standing public import path stable for scripts, saved notebooks, and
the Streamlit app during the staged refactor.
"""
from .domain.conditions import (
    ATOMIC_TOGGLE_KEYS,
    CONDITION_IDS,
    DEFAULT_CONDITION_STATES,
    active_condition_ids,
    params_to_text,
    parse_ints,
    parse_percent_list,
    parse_percent_ranges,
    strategy_narrative,
)

__all__ = [
    "ATOMIC_TOGGLE_KEYS", "CONDITION_IDS", "DEFAULT_CONDITION_STATES",
    "active_condition_ids", "params_to_text", "parse_ints",
    "parse_percent_list", "parse_percent_ranges", "strategy_narrative",
]
