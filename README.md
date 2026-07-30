# Natural-Language Alpha Research Agent

This is an incremental, hands-on AI-agent-engineering project. Sprint 1 implements one user-facing **Strategy Interpreter Agent**: it turns a natural-language trading rule into a constrained Strategy DSL v0.1, or returns precise clarification questions. It does not execute trades, calculate KPIs, or generate executable strategy code.

## Scope and architecture

The interpreter agent owns semantic interpretation and clarification because both require language judgment. It returns exactly one structured variant: `parsed` (a validated DSL plus explicit assumptions/warnings) or `clarification_required` (questions, original ambiguous terms, and an optional partial DSL).

The future Research Workflow is deliberately deterministic Python: it will validate data, calculate indicators/KPIs, simulate execution, apply costs, retain reproducibility, and enforce state transitions. LLMs must not calculate those values. Arbitrary Python generation and `eval` are rejected: the strategy is data, with explicit operands and conditions, not executable text.

The legacy `rhein/` application contains pre-existing backtesting code. It is intentionally not connected to `src/alpha_agent`, which is the isolated Sprint 1 learning module.

## DSL v0.1

`StrategyDefinition` pins `schema_version: "0.1"`, `frequency: "1d"`, `direction: "long_only"`, and `position_mode: "fully_invested_or_flat"`. It supports one symbol, one entry tree, and one exit tree.

- Market fields: `open`, `high`, `low`, `close`, `volume`
- Indicators: SMA, EMA, RSI, rolling return, rolling mean volume
- Conditions: persistent comparisons (`greater_than`, `less_than`, `greater_than_or_equal`, `less_than_or_equal`), cross events (`cross_above`, `cross_below`), and nested `and` / `or` groups
- Operands: current market fields, calculated indicators, finite scalars, and explicit scaled operands

For example, `volume > 1.5 × rolling_mean(volume, 20)` is a comparison whose right side is a `scaled_operand`; it is never flattened into text. A cross condition only permits two time-series operands, so it cannot be confused with `close > SMA(20)`.

Missing RSI windows are clarified by default. A future product policy can explicitly set `default_rsi_window`, at which point the interpreter must record that as an assumption.

DSL v0.2 additionally represents anchored, relative-day strategies: `t0` is an anchor day and `t1`, `t2`, and so on are explicit integer offsets. It supports anchor/entry-price references, date-ranged exits, and forced closes. Intraday triggers are represented only with an explicit `intraday_ohlcv` data requirement; a future daily-only backtest must reject them rather than fabricate an intraday path. Shorts, leverage, options, and portfolio allocation remain unsupported and must be rejected or clarified.

## Model and validation boundary

`StrategyInterpreterService` takes a typed `StrategyInterpretationRequest` and delegates to the provider-independent `StrategyModelClient` protocol. Provider adapters return JSON-compatible data only. The service then performs:

1. Pydantic schema/discriminated-union parsing.
2. Deterministic semantic validation, including operand compatibility, RSI ranges, non-empty groups, cross operand types, and indicator field constraints.
3. Typed errors for client failure, malformed model output, invalid schema, unsupported features, and semantic violations.

`FakeModelClient` queues structured responses or exceptions, so tests require no real external model.

## Monitoring the interpreter

Pass `InMemoryInterpreterMonitor` to the service to record one privacy-preserving event per attempt: outcome, latency, clarification/warning counts, error code, symbol presence, and hashes—not raw strategy text. `summary()` reports parse/clarification/failure rates, latency, and failure-code distribution.

Operational telemetry cannot prove interpretation quality. For that, attach reviewed examples through `record_evaluation(InterpretationEvaluation(...))`; the summary then reports reviewed-sample accuracy. This separates observed system health from evidence-backed quality evaluation and can later be replaced by a database or telemetry adapter without changing the interpreter.

## Development

Python 3.12+ is required.

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/unit/test_dsl.py tests/integration/test_interpreter.py
```

For a live local test, copy `.env.example` to `.env`, set `AWS_BEARER_TOKEN_BEDROCK`, and start the existing app with `streamlit run app.py`. The interpreter page reads the Bedrock key only from `.env`; it never displays or persists it. Open **Strategy Interpreter** in the sidebar, choose a group/symbol, enter a natural-language rule, then inspect the validated DSL or clarification response. `BEDROCK_MODEL` and `AWS_REGION` optionally override the displayed defaults. The page uses session-local, hashed monitoring events.

## Future sprints

Sprint 2 adds only the deterministic backtest workflow. Sprint 3 adds a hand-written research tool loop; Sprint 4 adds the evidence-based reviewer; Sprint 5 adds approval-gated immutable experiments; Sprint 6 adds state and memory; Sprint 7 may introduce LangGraph; Sprint 8 adds formal evaluation and production hardening.
