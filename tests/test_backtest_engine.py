from __future__ import annotations

from argparse import Namespace

import pandas as pd

from rhein.backtest import input_files, load_ohlc, load_profiles, resolve_strategy, run_backtest
from rhein.engine.kpis import calculate_kpis
from rhein.paths import CONFIG_ROOT, DATA_ROOT, GROUP_ROOT


def test_bundled_data_loads_and_runs() -> None:
    frame = load_ohlc(DATA_ROOT / "nvda.csv")
    trades, stats = run_backtest(frame)

    assert {"Date", "Open", "High", "Low", "Close", "Volume"} <= set(frame.columns)
    assert not trades.empty
    assert stats["n_trades"] == len(trades)
    assert stats["initial_capital"] == 10_000
    assert stats["final_equity"] > 0


def test_runner_accepts_the_t0_bullish_candle_toggle() -> None:
    frame = load_ohlc(DATA_ROOT / "nvda.csv")
    _, stats = run_backtest(frame, use_baseline_bullish_candle=False)
    assert stats["n_trades"] >= 0


def test_max_drawdown_includes_initial_capital_before_a_first_losing_trade() -> None:
    trades = pd.DataFrame({
        "ret_pct": [-5.0, 6.0, 15.0, 4.0],
        "pnl_eur": [-500.0, 570.0, 1500.0, 400.0],
        "days_held": [1, 1, 1, 1],
    })

    stats = calculate_kpis(trades, capital=10_000.0, compound=True)

    assert stats["max_drawdown_pct"] == -5.0


def test_runner_supports_t0_close_entry() -> None:
    frame = load_ohlc(DATA_ROOT / "nvda.csv")
    trades, stats = run_backtest(frame, entry_lag=0, hard_stop_days=(1, 2))

    assert stats["n_trades"] == len(trades)
    assert not trades.empty
    assert (trades["entry"] == trades["signal"]).all()


def test_group_directory_excludes_metadata_csvs() -> None:
    group = GROUP_ROOT / "03_高流动性_高波动"
    files = input_files(group)

    assert len(files) == 103
    assert all(path.name not in {"group_manifest.csv", "search_stage1_results.csv"} for path in files)


def test_profile_config_and_command_line_override() -> None:
    profiles = load_profiles(str(CONFIG_ROOT / "strategy_profiles.json"))
    args = Namespace(
        band_lo=0.02, band_hi=None, stop_pct=None, sma=None, entry_lag=None,
        entry_trend_fast_sma=None, entry_trend_slow_sma=None, hard_stop_days=None,
        cost_bps=None, close_stop=False,
    )

    strategy = resolve_strategy("NVDA", profiles, args)

    assert strategy["band_lo"] == 0.02
    assert strategy["band_hi"] > strategy["band_lo"]
    assert all(day > strategy["entry_lag"] for day in strategy["hard_stop_days"])
