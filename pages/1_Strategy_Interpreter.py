"""Local Streamlit page for manually testing the Sprint 1 interpreter."""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

import streamlit as st
from dotenv import load_dotenv
from rhein.data.discovery import input_files
from rhein.ui.presets import available_data_scopes
from alpha_agent.domain.interpretation import StrategyInterpretationRequest
from alpha_agent.errors import AlphaAgentError
from alpha_agent.model.bedrock_client import BedrockStrategyModelClient
from alpha_agent.monitoring import InMemoryInterpreterMonitor
from alpha_agent.parser.service import StrategyInterpreterService
from alpha_agent.strategy_library import JsonStrategyLibrary, SavedStrategy, StrategyLibraryError

# The project-local file is the explicit source of truth for this local app.
load_dotenv(ROOT / ".env", override=True)

st.set_page_config(page_title="自然语言策略解释器", layout="wide")
st.title("自然语言策略解释器")
st.caption("Sprint 1：将策略描述转换为 DSL v0.1，或提出精准澄清问题；本页面不会执行回测。")

if "interpreter_monitor" not in st.session_state: st.session_state.interpreter_monitor = InMemoryInterpreterMonitor()
if "interpreter_strategy_text" not in st.session_state:
    st.session_state.interpreter_strategy_text = "当收盘价向上穿越 20 日均线时买入；当它向下穿越该均线时卖出。"


def load_strategy_text(text: str) -> None:
    st.session_state.interpreter_strategy_text = text


library = JsonStrategyLibrary(ROOT / "config" / "saved_strategies.json")
with st.sidebar:
    st.subheader("模型配置")
    model = st.text_input("Bedrock 模型", value=os.getenv("BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-6"))
    region = st.text_input("AWS 区域", value=os.getenv("AWS_REGION", "us-east-1"))
    st.caption("API 密钥仅从项目根目录 `.env` 的 `AWS_BEARER_TOKEN_BEDROCK` 读取，不在页面显示或保存。")

scope_options = available_data_scopes()
scope_labels = list(scope_options)
main_scope = st.session_state.get("data_scope")
default_scope_index = scope_labels.index(main_scope) if main_scope in scope_options else 0
selected_scope = st.selectbox(
    "标的分组",
    scope_labels,
    index=default_scope_index,
    key="data_scope",
    help="与主回测页面共享同一个数据范围和 Nasdaq 分组选择。",
)
if selected_scope == "自定义路径":
    data_path = st.text_input("自定义数据目录或 CSV", value="data", key="interpreter_custom_path")
else:
    data_path = scope_options[selected_scope]
    st.caption(f"当前数据范围：`{data_path}`")
try:
    group_symbols = [path.stem.upper() for path in input_files(Path(data_path))]
except ValueError as error:
    st.error(f"无法加载此分组的标的：{error}")
    group_symbols = []
if group_symbols:
    symbol = st.selectbox(
        "标的",
        group_symbols,
        index=group_symbols.index("AAPL") if "AAPL" in group_symbols else 0,
        key="interpreter_symbol",
        help="仅显示当前所选组（或数据范围）中的标的。",
    )
else:
    symbol = None
strategy_text = st.text_area("用自然语言描述日频、仅做多策略", height=150, key="interpreter_strategy_text")
if st.button("解释策略", type="primary"):
    if not symbol:
        st.error("先选择一个包含有效 OHLC 数据的标的分组。")
    elif not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
        st.error("未找到 API 密钥。请在项目根目录 `.env` 中设置 AWS_BEARER_TOKEN_BEDROCK 后重试。")
    else:
        service = StrategyInterpreterService(BedrockStrategyModelClient(model=model, region=region), monitor=st.session_state.interpreter_monitor)
        try:
            with st.spinner("正在解释策略…"):
                result = service.interpret(StrategyInterpretationRequest(strategy_text=strategy_text, symbol=symbol))
            if result.status == "parsed":
                st.success("策略已解析并通过语义验证。")
                st.json(result.model_dump(mode="json"))
                st.session_state["last_parsed_strategy"] = result
                st.session_state["last_strategy_text"] = strategy_text
            else:
                st.warning("需要补充澄清信息。")
                st.json(result.model_dump(mode="json"))
        except AlphaAgentError as error:
            st.error(f"{error.code}: {error.message}")
            if error.code == "model_client_failure":
                st.info("这是模型服务或模型配置问题，不表示策略本身已被 DSL 拒绝。请查看下方详情中的 provider_message，并确认侧边栏模型对当前 API 项目可用。")
            with st.expander("机器可读的错误详情"): st.json(error.details)

last_result = st.session_state.get("last_parsed_strategy")
matching_result = last_result if st.session_state.get("last_strategy_text") == strategy_text else None
st.subheader("保存为跨组策略")
st.caption("始终保存原始用户语言；当前文本若已成功解析，则同时保存已验证 DSL。策略不绑定当前分组。")
strategy_name = st.text_input("策略名称", value=(matching_result.strategy.strategy_name if matching_result is not None and matching_result.strategy.strategy_name else "未命名策略"), key="saved_strategy_name")
if matching_result is None:
    st.info("当前文本尚未成功解析；可以保存为待解释策略草稿。")
if st.button("保存策略", type="secondary"):
    try:
        saved = library.save(SavedStrategy(
            strategy_name=strategy_name,
            original_language=strategy_text,
            strategy=matching_result.strategy if matching_result is not None else None,
            assumptions=matching_result.assumptions if matching_result is not None else [],
            warnings=matching_result.warnings if matching_result is not None else [],
        ))
        suffix = "（已附带验证过的 DSL）" if saved.strategy is not None else "（待解释草稿，尚无 DSL）"
        st.success(f"已保存跨组策略：{saved.strategy_name}{suffix}")
    except StrategyLibraryError as error:
        st.error(f"{error.code}: {error.message}")

st.subheader("已保存的跨组策略")
try:
    saved_strategies = library.list()
    if saved_strategies:
        saved_ids = {str(item.strategy_id): item for item in saved_strategies}
        selected_saved_id = st.selectbox("选择策略", list(saved_ids), format_func=lambda item: f"{saved_ids[item].strategy_name} · {saved_ids[item].created_at:%Y-%m-%d %H:%M}")
        selected_saved = saved_ids[selected_saved_id]
        st.caption("原始描述：" + selected_saved.original_language)
        st.button("加载原始策略到输入框", key="load_saved_strategy", on_click=load_strategy_text, args=(selected_saved.original_language,))
        if selected_saved.strategy is None:
            st.info("此策略保存时尚未形成 DSL，属于待解释草稿。")
        else:
            with st.expander("查看已保存 DSL"):
                st.json(selected_saved.strategy.model_dump(mode="json"))
    else:
        st.caption("还没有已保存的跨组策略。")
except StrategyLibraryError as error:
    st.error(f"无法读取策略库：{error.message}")

st.subheader("解释器监控（当前浏览器会话）")
st.json(st.session_state.interpreter_monitor.summary())
st.caption("监控仅保存输入哈希，不保存策略原文。完成人工或离线评估后才会显示质量准确率；运行指标本身不等于解释准确率。")
