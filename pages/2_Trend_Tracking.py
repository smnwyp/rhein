"""Independent Streamlit page for deterministic sector-emergence research."""
from __future__ import annotations

import sys
import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st

from rhein.ui.presets import available_data_scopes
import rhein.trend_tracking as _trend_tracking_core
import rhein.ui.trend_tracking as _trend_tracking

# A long-running local Streamlit process caches imported page helpers. Reload
# the deterministic core before its UI helper so their function signatures stay
# in sync after a local code update; research state stays in session data.
_trend_tracking_core = importlib.reload(_trend_tracking_core)
_trend_tracking = importlib.reload(_trend_tracking)
render_trend_tracking = _trend_tracking.render_trend_tracking


st.set_page_config(page_title="板块趋势追踪", layout="wide")

scope_options = available_data_scopes()
scope_labels = list(scope_options)
default_scope = "全部 Nasdaq 当前股票池"
with st.sidebar:
    st.header("数据范围")
    selected_scope = st.selectbox(
        "标的分组",
        scope_labels,
        index=scope_labels.index(st.session_state["trend_tracking_data_scope"])
        if st.session_state.get("trend_tracking_data_scope") in scope_options else scope_labels.index(default_scope),
        key="trend_tracking_data_scope",
        help="板块日度指标只使用当前范围中的日线标的。",
    )
    if selected_scope == "自定义路径":
        data_path = st.text_input("自定义数据目录或 CSV", value="data", key="trend_tracking_custom_path")
    else:
        data_path = scope_options[selected_scope]
        st.caption(f"当前数据范围：`{data_path}`")

render_trend_tracking(data_path=data_path, scope_label=selected_scope)
