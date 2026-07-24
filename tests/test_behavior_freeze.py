from __future__ import annotations

from rhein.backtest import load_ohlc, run_backtest
from rhein.paths import DATA_ROOT


def test_nvda_default_trade_and_kpi_snapshot() -> None:
    """冻结重构前的默认策略交易序列与 KPI，防止“结果接近”掩盖行为漂移。"""
    trades, stats = run_backtest(load_ohlc(DATA_ROOT / "nvda.csv"), compound=False)

    assert trades.iloc[:3].to_dict("records") == [
        {"signal": "2010-08-25", "entry": "2010-08-27", "exit": "2010-08-30", "days_held": 1,
         "entry_px": 0.2317, "exit_px": 0.227, "ret_pct": -2.0, "pnl_eur": -200.0, "reason": "t3 日内止损"},
        {"signal": "2012-06-15", "entry": "2012-06-19", "exit": "2012-06-21", "days_held": 2,
         "entry_px": 0.3031, "exit_px": 0.297, "ret_pct": -2.0, "pnl_eur": -200.0, "reason": "t4 日内止损"},
        {"signal": "2015-04-09", "entry": "2015-04-13", "exit": "2015-04-16", "days_held": 3,
         "entry_px": 0.5425, "exit_px": 0.5411, "ret_pct": -0.266, "pnl_eur": -26.61, "reason": "收盘价低于入场价"},
    ]
    assert {key: stats[key] for key in ("n_trades", "final_equity", "win_rate_pct", "payoff_ratio", "max_drawdown_pct")} == {
        "n_trades": 15, "final_equity": 11584.49, "win_rate_pct": 40.0,
        "payoff_ratio": 3.126, "max_drawdown_pct": -4.21,
    }


def test_test_period_never_counts_pre_split_signals() -> None:
    df = load_ohlc(DATA_ROOT / "nvda.csv")
    split_date = df["Date"].iloc[int(len(df) * 0.7)]
    trades, _ = run_backtest(
        df, signal_start=split_date, compound=False,
        use_signal_band=False, use_baseline_prior_low=False, use_baseline_max_rise=False,
        use_baseline_rsi=False, use_baseline_close_above_fast_sma=False,
        use_baseline_fast_above_slow_sma=False, use_baseline_close_above_sma20=False,
        use_baseline_volume_sma=False, use_entry_close_vs_t0=False,
        use_early_stop=False, use_exit_below_entry=True, use_exit_below_sma=False,
        use_forced_exit=True, forced_exit_day=5, use_forced_exit_intraday_protection=False,
    )
    assert not trades.empty
    assert min(trades["signal"]) >= str(split_date.date())
