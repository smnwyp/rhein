# NVDA / TSLA 参数档案使用说明

`rhein.backtest` 保持同一个 momentum breakout 策略结构，但会按 CSV 文件名读取 `config/strategy_profiles.json` 中对应的参数。`nvda.csv` 使用 `NVDA` 档案，`tesla.csv` 使用 `TESLA` 档案。

## 时间标记

- `t0`：信号日。
- `entry_lag`：确认并在何日入场。`1` 是 t1，`2` 是 t2，`3` 是 t3。
- `hard_stop_days`：相对 t0 的早期止损检查日，且必须晚于入场日。例如 t2 入场通常配 `3,4`；若改为 t3 入场，应相应使用 `4,5`。
- 所有条件已原子化，默认全部开启：信号涨幅区间；最低点不在 t0；最大延伸；RSI；tN≥t0；确认窗口的收盘>快线、快线>慢线、短均量>长均量；早期止损；后期跌破入场价；后期跌破 SMA。UI 中每条都有独立开关。
- 入场窗口只考察 tN-1、tN，绝不使用 tN+1。启用的入场窗口条件必须在同一天同时成立；默认快/慢线为 SMA5/SMA10、量均线为 5/20。

## 当前初始档案

| 标的 | 信号涨幅 | 入场确认 | 早期止损日 | 止损 | 原因 |
|---|---:|---:|---:|---:|---|
| NVDA | 2.0%–2.5% | t2 | t3、t4 | 2% | 保留历史基准规则。|
| TSLA | 2.0%–3.0% | t2 | t3、t4 | 3% | TSLA 日内波动更大，较宽的信号带与止损可减少被 2% 噪声止损的情况。|

这是基于现有数据的**初始假设**，不是已经样本外证明的最优参数。调整后应同时比较胜率、盈利因子和最大回撤。

## 常用命令

使用两个标的各自档案：

```bash
.venv/bin/python -m scripts.backtest data
```

测试所有标的改为 t3 确认、t4/t5 早期止损（命令行参数会覆盖档案）：

```bash
.venv/bin/python -m scripts.backtest data --entry-lag 3 --hard-stop-days 4,5
```

测试统一更宽的 2%–3% 信号带与 3% 止损：

```bash
.venv/bin/python -m scripts.backtest data --band-lo 0.02 --band-hi 0.03 --stop-pct 0.03
```

仅测试一个标的，避免把参数意外覆盖到另一只股票：

```bash
.venv/bin/python -m scripts.backtest data/tesla.csv --entry-lag 3 --hard-stop-days 4,5
```

每次输出的 Markdown 报告都会在每个标的标题下列出“实际策略参数”，它才是该次结果的准确口径。

## 参数组合扫描

`config/parameter_grid.json` 定义每只股票要枚举的信号区间、确认日与止损比例。早期止损日会随确认日自动对应：t1 入场检查 t2/t3，t2 入场检查 t3/t4，t3 入场检查 t4/t5。

```bash
.venv/bin/python -m scripts.backtest data --sweep
```

扫描会生成完整 Markdown 表格、CSV 与 JSON。默认使用复利模式；加入 `--no-compound` 可改用固定仓位模式。扫描结果仅是样本内比较，不能直接把排名第一的组合当成最终实盘参数。
