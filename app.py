"""Streamlit Cloud entry point.

The implementation lives in :mod:`rhein.ui_app`; keeping this thin file at the
repository root preserves the configured Streamlit command: ``streamlit run app.py``.
"""

# Streamlit entrypoint. 不能用 ``from rhein.ui_app import *``：Python 会缓存
# 该模块，后续交互 rerun 不会重新执行 UI，因而呈现空白页。run_module 则每次执行。
import runpy

runpy.run_module("rhein.ui_app", run_name="__main__")
