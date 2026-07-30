# Natural-Language Alpha Research Agent

This repository is being extended incrementally as a hands-on agent-engineering project. Sprint 1 adds a provider-independent **Strategy Interpreter Agent** that turns natural language into a constrained Strategy DSL v0.1 or asks structured clarification questions. It does not generate or execute Python.

## Core architecture

1. **Strategy Interpreter Agent** — interprets intent and ambiguity in one task, returning `ParsedStrategy` or `ClarificationRequired`.
2. **Research Workflow** — will deterministically validate and execute strategies, backtests, KPIs, robustness checks, and reports.
3. **Research Reviewer Agent** — will review standardized evidence and propose bounded experiments.
4. **Experiment Workflow** — will apply only approved changes, version strategies, rerun research, and compare results.

Only item 1 is implemented in Sprint 1. Semantic validation is deterministic Python and is kept separate from both prompts and Pydantic's structural validation. Existing `rhein` backtest code is not connected to the interpreter.

## DSL v0.1

The immutable, strict Pydantic models support OHLCV fields; SMA, EMA, RSI, rolling return, and rolling mean volume; persistent `greater_than`/`less_than` comparisons; event-based `cross_above`/`cross_below`; and nested `and`/`or` groups. A narrowly scoped scaled-series operand represents rules such as `volume > 1.5 × rolling_mean_volume(20)` without introducing general arithmetic.

Every strategy explicitly records `schema_version: "0.1"`, daily frequency, one asset, long-only direction, and fully-invested-or-flat sizing. Crosses require two time series. The validator checks operand dimensions, recursive groups, and RSI threshold ranges.

```json
{
  "status": "parsed",
  "strategy": {
    "schema_version": "0.1",
    "asset": "AAPL",
    "frequency": "daily",
    "direction": "long_only",
    "position_sizing": "fully_invested_or_flat",
    "entry": {
      "kind": "cross",
      "operator": "cross_above",
      "left": {"kind": "market_field", "field": "close"},
      "right": {"kind": "indicator", "indicator": {"type": "sma", "source": "close", "window": 20}}
    },
    "exit": {
      "kind": "cross",
      "operator": "cross_below",
      "left": {"kind": "market_field", "field": "close"},
      "right": {"kind": "indicator", "indicator": {"type": "sma", "source": "close", "window": 20}}
    }
  },
  "assumptions": [{"code": "ma_is_sma", "message": "Moving average interpreted as SMA."}],
  "warnings": []
}
```

Ambiguous phrases are preserved verbatim. For example, “fallen significantly” and “volume is increasing” produce questions asking for measurable definitions rather than guessed windows or thresholds.

## Model boundary

`ModelClient` is a small protocol that accepts system/user prompts and returns JSON-compatible structured data. `StrategyInterpreter` validates that data against the discriminated result union, then applies deterministic semantic validation. `FakeModelClient` supplies queued responses for network-free tests; a provider adapter can be added later without changing domain models or orchestration.

## Development

Python 3.12 or newer is required.

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

The tests cover all five required examples, nested logic, malformed DSL, unsupported indicators/frequencies, invalid windows/RSI thresholds/cross operands, result discrimination, JSON round trips, and schema-version preservation.

## Not in Sprint 1

There is no model-provider adapter, LangChain/LangGraph integration, indicator calculation, signal execution, backtesting, KPI calculation, review agent, or experiment workflow in this sprint.
