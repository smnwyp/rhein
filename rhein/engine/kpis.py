"""KPI calculations shared by the backtest runner and reporting layer."""
import pandas as pd


def calculate_kpis(trades: pd.DataFrame, capital: float, compound: bool) -> dict:
    """计算报告中的 KPI；金额均以输入资本的计价货币表示。"""
    mode = "复利" if compound else "固定仓位"
    if trades.empty:
        return {"mode": mode, "n_trades": 0, "initial_capital": capital,
                "final_equity": capital, "total_pnl": 0.0}
    returns = trades["ret_pct"] / 100
    pnl = trades["pnl_eur"]
    winners, losers = returns > 0, returns <= 0
    equity = (capital * (1 + returns).cumprod() if compound
              else capital + pnl.cumsum())
    drawdown = equity / equity.cummax() - 1
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl <= 0].sum())
    avg_win = float(returns[winners].mean()) if winners.any() else None
    avg_loss = float(returns[losers].mean()) if losers.any() else None
    payoff_ratio = (avg_win / abs(avg_loss) if avg_win is not None and avg_loss not in (None, 0)
                    else None)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        "mode": mode, "n_trades": int(len(trades)),
        "initial_capital": round(capital, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_pnl": round(float(equity.iloc[-1] - capital), 2),
        "win_rate_pct": round(float(winners.mean() * 100), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 3),
        "avg_win_pct": round(avg_win * 100, 3) if avg_win is not None else None,
        "avg_loss_pct": round(avg_loss * 100, 3) if avg_loss is not None else None,
        "payoff_ratio": round(payoff_ratio, 3) if payoff_ratio is not None else None,
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "max_drawdown_pct": round(float(drawdown.min() * 100), 2),
        "avg_days_held": round(float(trades["days_held"].mean()), 2),
        "max_win_pct": round(float(returns.max() * 100), 3),
        "max_loss_pct": round(float(returns.min() * 100), 3),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
    }


def cross_asset_summary(results: list[dict], mode: str) -> dict:
    """将独立标的的交易合并为比较统计，不把它误当成可交易组合。"""
    selected = [r for r in results if r["stats"]["mode"] == mode]
    frames = [r["trades"] for r in selected if not r["trades"].empty]
    initial = sum(r["stats"]["initial_capital"] for r in selected)
    final = sum(r["stats"]["final_equity"] for r in selected)
    if not frames:
        return {"mode": mode, "n_trades": 0, "initial": initial, "final": final,
                "pnl": final - initial, "win_rate": None, "payoff": None, "pf": None}
    trades = pd.concat(frames, ignore_index=True)
    returns, pnl = trades["ret_pct"] / 100, trades["pnl_eur"]
    winners, losers = returns > 0, returns <= 0
    avg_win = returns[winners].mean() if winners.any() else None
    avg_loss = returns[losers].mean() if losers.any() else None
    gross_profit, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {"mode": mode, "n_trades": len(trades), "initial": round(initial, 2),
            "final": round(final, 2), "pnl": round(final - initial, 2),
            "win_rate": round(float(winners.mean() * 100), 2),
            "payoff": round(float(avg_win / abs(avg_loss)), 3)
            if avg_win is not None and avg_loss not in (None, 0) else None,
            "pf": round(float(gross_profit / gross_loss), 3) if gross_loss > 0 else None}
