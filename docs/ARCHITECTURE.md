# Codebase architecture

## Entrypoints

- `app.py` — Streamlit Cloud entrypoint; delegates to `rhein.ui_app`.
- `app.py` is the sole root-level application entrypoint for Streamlit Cloud.
- CLI commands live in `scripts/` and can be invoked with `python -m scripts.<name>`.

## Application package: `rhein/`

- `backtest.py` — OHLC loading, strategy execution, KPIs, reporting, and CLI implementation.
- `strategy.py` — atomic-condition IDs and parameter parsing.
- `paths.py` — repository-relative data and report paths.
- `ui_app.py` — Streamlit composition and interactive workflow.

## Operational scripts: `scripts/`

- Data acquisition: `download_nasdaq.py`
- Feature groups: `build_nasdaq_feature_groups.py`, `organize_nasdaq_groups.py`
- Research and scans: `scan_all_group_domains.py`, `scan_high_volatility_group.py`,
  `analyze_group_statistics.py`, `analyze_group_symbol_quality.py`

Run the canonical module form, for example:

```bash
.venv/bin/python -m scripts.scan_all_group_domains
```

All internal imports should target `rhein.*`; command-line behavior belongs in `scripts/`.
