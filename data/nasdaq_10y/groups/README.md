# Nasdaq 特征分组数据

每个目录中的股票 CSV 是指向 `data/nasdaq_10y/` 原始文件的软链接，不重复占用数据空间。
可直接将任意组目录作为 `backtest.py` 或 UI 的数据目录。

分组规则详见项目根目录的 `SEARCH_DOMAIN_AND_FILTERS.md`；每组的 `group_manifest.csv` 包含对应特征值。
