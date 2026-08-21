# Project Handoff

Last updated: 2026-08-18

## Current branch and release

- Branch: `feature/poc`
- `HEAD` / `origin/feature/poc`: `a8ebfbdc` — `Bootstrap A-share data in deployed UI`
- A-share release: [a-share-data-2026-08-12](https://github.com/smnwyp/rhein/releases/tag/a-share-data-2026-08-12)
- Data asset: [`a_share_ohlcv.tar.gz`](https://github.com/smnwyp/rhein/releases/download/a-share-data-2026-08-12/a_share_ohlcv.tar.gz) (153,105,981 bytes)
- SHA-256: `c6dcbd94f5ee4e1d2f95fa57cecedc3dc73e80e50bf1a13f1bcc58278d83118e`

## A-share data workflow

1. Source TXT exports are local-only under `data/A股/`.
2. `scripts/convert_a_share_ohlcv.py` converts the TongDaXin exports to project-standard OHLCV CSV files in `data/a_share_ohlcv/`.
3. The current conversion report identifies 5,466 usable CSVs. 74 source files are excluded: 9 contain no bars and 65 have non-positive adjusted prices, which are unsafe for return-based backtests.
4. Market-data directories and `artifacts/` are ignored by Git. Do not re-add the complete data universe to Git.
5. `scripts/package_a_share_data.py` creates the release archive; `scripts/download_a_share_data.py` downloads, validates, and installs it.

## Deployed UI behavior

- `rhein/ui_app.py`, `pages/1_Strategy_Interpreter.py`, and `pages/2_Trend_Tracking.py` call `ensure_a_share_data()` before presenting data scopes.
- On a fresh Streamlit runtime, it downloads the public release asset, validates SHA-256, installs `data/a_share_ohlcv/`, and then the "全部 A 股" scope appears.
- First startup downloads about 146 MB and can take a few minutes. Subsequent starts use the `.rhein_a_share_package_ready` marker.
- Override the published asset without code changes using environment variables:
  - `A_SHARE_DATA_URL`
  - `A_SHARE_DATA_SHA256`

## Validation completed

- Conversion output: 5,466 OHLCV CSVs and 10,674,574 daily rows loaded successfully through the project loader.
- Package archive extraction was tested with the full archive.
- Latest related test run: `27 passed`.

## Working tree: do not commit without review

Three staged files are currently present but are not part of the A-share delivery. They appear to be duplicate copies with a ` 2` suffix:

- `rhein/data/a_share 2.py`
- `scripts/convert_a_share_ohlcv 2.py`
- `tests/test_a_share_converter 2.py`

Review and either intentionally rename/merge them or remove them from the index before the next commit. They were deliberately excluded from the A-share commits and pushes.

## Git convention

Only execute `git push` when the user says **`bibu`**. Without that passphrase, commits may be made when required, but do not push.
