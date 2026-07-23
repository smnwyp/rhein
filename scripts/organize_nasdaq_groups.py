"""按 nasdaq_feature_groups.csv 为 CSV 数据创建分组目录软链接。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/nasdaq_10y")
GROUP_DIR = DATA_DIR / "groups"
GROUP_NAMES = {
    "高流动性－低波动": "01_高流动性_低波动",
    "高流动性－中波动": "02_高流动性_中波动",
    "高流动性－高波动": "03_高流动性_高波动",
    "中流动性－低波动": "04_中流动性_低波动",
    "中流动性－中波动": "05_中流动性_中波动",
    "中流动性－高波动": "06_中流动性_高波动",
    "低流动性－低波动": "07_低流动性_低波动",
    "低流动性－中波动": "08_低流动性_中波动",
    "低流动性－高波动": "09_低流动性_高波动",
    "历史不足1年（不进入主验证）": "10_历史不足1年",
}


def main() -> None:
    groups = pd.read_csv(DATA_DIR / "nasdaq_feature_groups.csv")
    GROUP_DIR.mkdir(exist_ok=True)
    readme = """# Nasdaq 特征分组数据

每个目录中的股票 CSV 是指向 `data/nasdaq_10y/` 原始文件的软链接，不重复占用数据空间。
可直接将任意组目录作为 `backtest.py` 或 UI 的数据目录。

分组规则详见项目根目录的 `SEARCH_DOMAIN_AND_FILTERS.md`；每组的 `group_manifest.csv` 包含对应特征值。
"""
    (GROUP_DIR / "README.md").write_text(readme, encoding="utf-8")
    linked = 0
    for group_name, subset in groups.groupby("策略特征组", sort=False):
        folder_name = GROUP_NAMES.get(group_name)
        if folder_name is None:
            raise ValueError(f"未定义目录名称的分组：{group_name}")
        folder = GROUP_DIR / folder_name
        folder.mkdir(exist_ok=True)
        subset.to_csv(folder / "group_manifest.csv", index=False)
        for symbol in subset["symbol"]:
            source = (DATA_DIR / f"{symbol}.csv").resolve()
            destination = folder / f"{symbol}.csv"
            if not source.is_file():
                raise FileNotFoundError(f"缺少原始 CSV：{source}")
            if destination.is_symlink() and destination.resolve() == source:
                continue
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"目标已存在且不是预期链接：{destination}")
            destination.symlink_to(source)
            linked += 1
        print(f"{folder_name}：{len(subset)} 个标的")
    print(f"完成：本次新增 {linked} 个软链接；目录：{GROUP_DIR}")


if __name__ == "__main__":
    main()
