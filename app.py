"""Streamlit Cloud entry point.

The implementation lives in :mod:`rhein.ui_app`; keeping this thin file at the
repository root preserves the configured Streamlit command: ``streamlit run app.py``.
"""

"""Streamlit entrypoint.

不要用 ``from rhein.ui_app import *``：Streamlit 的每次交互都会重跑这个文件，
但 Python 会缓存被 import 的模块，导致第二次运行不再执行 UI 代码并呈现空白页。
run_module 会在每一次 rerun 都执行 UI 模块。
"""
import runpy

runpy.run_module("rhein.ui_app", run_name="__main__")
