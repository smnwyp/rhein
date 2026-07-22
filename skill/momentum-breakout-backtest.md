---
name: momentum-breakout-backtest
description: Backtest Chloe's daily-candle momentum breakout strategy ("sharp rise, ride the move") exactly as specified, with the bundled engine. Use this skill whenever she asks to backtest, re-run, tweak, sweep, or ablate this strategy, mentions t0/t2 entry, the 2–2.5% signal band, the t3/t4 stop, the SMA5 exit, or asks to run her strategy on some ticker or CSV. Also use it when she proposes rule changes — modify parameters or the engine rather than rewriting from scratch, so results stay comparable across sessions.
---

# Momentum Breakout Backtest

Owner: Chloe. Version: v1.8 (2026-07-22: early hard stop is now always close-triggered and filled at the close; all legacy defaults remain unchanged). Instrument scope: single instrument, one position at a time, long only, daily OHLCV candles. Capital: EUR 10,000. Baseline single-side cost: 0 bps.

## Strategy specification (canonical — do not silently reinterpret)

## Condition IDs

| ID | Condition |
|---|---|
| T0-01 | Signal daily-return band |
| T0-02 | The lowest close in the baseline window is not t0 |
| T0-03 | t0's extension above that low is capped |
| T0-04 | t0 RSI ceiling |
| EN-01 | tN close is at least t0 close |
| EN-02 | Confirmation-window close is above fast SMA |
| EN-03 | Confirmation-window fast SMA is above slow SMA |
| EN-04 | Confirmation-window volume short SMA is above volume long SMA |
| EX-01 | Early hard stop |
| EX-02 | Later close below entry-price exit |
| EX-03 | Later close below exit-SMA exit |
| EX-04 | Force close at a specified t0-relative day (default t5) |
| EX-05 | On the EX-04 day, intraday protection below entry by a configured amount (default 1%) |

These identifiers are stable labels for the UI, notes and experiments; the corresponding engine flags remain `use_*` parameters.

**Signal and baseline (t0):** the following are four independent conditions, all enabled by default: (1) `close[t0] / close[t0-1] − 1` is within [2.0%, 2.5%] (band, not open-ended); (2) in the inclusive window `t0-15 … t0`, the lowest close (`t-min`) must not be t0; (3) `close[t0] ≤ 1.20 × close[t-min]`; (4) Wilder RSI(14) at t0 is ≤90. Any disabled condition is not evaluated for eligibility. Lookback, extension, RSI period and RSI ceiling remain configurable.

**Entry confirmation (default t2):** `entry_lag` trading days after t0. The base condition `close[tN] >= close[t0]` is independently switchable and enabled by default.

**Entry conditions (all independently switchable, enabled by default):** inspect only `tN-1` and `tN`; never use tN+1. For any enabled subset, **any one** of those days must meet every enabled condition:

1. `close[d] > SMA_fast[d]`;
2. `SMA_fast[d] > SMA_slow[d]`;
3. `VolumeSMA_short[d] > VolumeSMA_long[d]`.

Defaults are price `SMA_fast = 5`, `SMA_slow = 10`, and volume `SMA_short = 5`, `SMA_long = 20`; all four periods are configurable and each fast/slow pair must be ordered short < long.

Buy at `close[tN]` if the base confirmation and this filter pass; otherwise skip. This uses no future-day information.

**Exit — early hard stop:** independently switchable, enabled by default. On configured `hard_stop_days` relative to t0 (default for t2 is t3/t4), if `close <= (1 − stop_pct)·P`, exit at that day's close. Intraday lows and opening gaps do not trigger this rule.

**Exit — after the early-stop window,** at the close, first condition hit wins:
1. `close < P` (close below entry), independently switchable; or
2. `close < exit_SMA` (default simple SMA5 of closes, including the current day; this exit SMA is separate from the entry fast/slow SMAs), independently switchable.

If both later exits are off, the position stays open until the data ends (unless the early stop is enabled and triggered).

**Exit — forced holding-period cap (EX-04, off by default):** force the position closed at the selected t0-relative day, default `t5`. Therefore a default t2 entry is closed at the t5 close. The selected day must be later than the entry day. If the early stop is triggered on that same day, the early stop takes precedence because it can occur intraday. This is intended as an explicit fixed-holding-period benchmark.

**Forced-exit-day intraday protection (EX-05, enabled by default when EX-04 is used):** on the EX-04 day only, if any intraday price touches `entry_price × (1 − protection_pct)` (default `1%`), exit immediately at the protection level; if the day opens below it, exit at the open. With daily OHLC data, `low ≤ protection level` is the observable proxy for that touch. If it does not trigger, EX-04 exits at that day's close. EX-05 is independently switchable.

**Portfolio logic:** signals occurring while a position is open are ignored. After an exit, scanning resumes the next day. Default mode is **fixed stake** (EUR 10k per trade, non-compounding); compounding is an explicit alternative. Multi-instrument results remain independent-account comparisons, not a capital-allocated portfolio backtest.

## How to run

Engine: `backtest.py` (pandas/numpy only, no network needed).

```bash
.venv/bin/python backtest.py data.csv
.venv/bin/python backtest.py data.csv --close-stop
.venv/bin/python backtest.py data.csv --entry-trend-fast-sma 8 --entry-trend-slow-sma 20
.venv/bin/python backtest.py data.csv --stop-pct 0.03 --sma 10 --capital 10000 --compound
```

Data format: CSV with `Date, Open, High, Low, Close, Volume`, daily, oldest→newest (loader sorts anyway). The sandbox cannot reach Yahoo Finance; either the user uploads a CSV, or she runs locally: `yf.download("^GDAXI", start="2010-01-01").to_csv("dax.csv")`.

For sweeps/ablations, import instead of shelling out:

```python
from backtest import load_ohlc, run_backtest
df = load_ohlc("data.csv")
trades, stats = run_backtest(df, band_lo=0.02, band_hi=0.025, stop_pct=0.02,
                             sma_n=5, capital=10_000, compound=True,
                             stop_intraday=False, entry_trend_fast_sma=5,
                             entry_trend_slow_sma=10)
```

## Reporting standard (what a run must include)

1. State the mode used (fixed stake by default) and report n_trades, win rate, avg win/loss %, avg days held, total PnL, final equity, max drawdown.
2. **Exit-reason distribution** (`trades["reason"].value_counts()`) — this is the main diagnostic. Known baseline: on a synthetic random walk the strategy is ~break-even with "close < entry" as the dominant exit; real-data results should be judged against that null, not against zero.
3. If she asks "does it work": check trade count first. Fewer than ~30 trades → say the sample is too small for a verdict, suggest widening the band or extending history.

## Known design tensions (raise when relevant, don't nag)

- The `close < P` exit after the early-stop window converts small winners into scratches and caps the "ride the move" upside; the natural ablation is dropping rule (1) and keeping only the exit-SMA rule.
- The [2%, 2.5%] band is narrow → few signals. The `--band-hi 999` variant is the standard comparison.
- Multi-instrument scanning (many stocks, capital allocation) is out of scope for this engine — flag as a separate build if she asks.

## Change discipline

When she changes a rule, bump the version line above, note the change in one line, and keep old parameter defaults reachable via CLI flags so v1 results remain reproducible. Every condition change must be represented as an explicit boolean `use_*` parameter in the engine and a corresponding UI toggle; never combine multiple conditions behind one switch.
