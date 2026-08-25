# 后台回测与 Parquet 迁移

## 运行模型

- Streamlit 只创建并读取回测 job；不在网页请求中逐标的执行。
- `python -m rhein.backtest_worker --project-root . --job-id <UUID>` 在独立进程运行。
- Job 状态在 `config/backtest_jobs/<job>.json`，每个标的的 KPI 与交易账本在独立结果文件中。页面关闭、刷新或 worker 重启后，点击“继续后台回测”会跳过已完成标的。
- 成功后 worker 将不可变结果写入既有的 `saved_backtest_artifacts/` 历史结构；复盘仍只读取保存的回测。

## 数据格式

- `load_ohlc()` 优先支持 Parquet，同时保留 CSV 兼容。
- 数据发现对同一标的优先选 `.parquet`，不会因迁移期两种文件并存而重复回测。
- 转换命令：

```bash
.venv/bin/python scripts/convert_a_share_ohlcv.py --input-path data/A股 --output-dir data/a_share_ohlcv --format parquet
```

- 发布包命令：

```bash
.venv/bin/python scripts/package_a_share_data.py --source-dir data/a_share_ohlcv --output artifacts/a_share_ohlcv_parquet.tar.gz
```

本地发布候选包不应提交进 Git。将它上传为新的数据 asset 后，才可把 Streamlit 部署的 `A_SHARE_DATA_URL` 与 `A_SHARE_DATA_SHA256` 指向该 asset；在此之前，线上旧 CSV asset 仍可正常兼容读取。
