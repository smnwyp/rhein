"""从已有 Nasdaq 分类快照生成细粒度统计研究板块映射。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RESEARCH_COLUMNS = ["代码", "行业板块", "Nasdaq板块", "Nasdaq细分行业", "数据源", "分类快照UTC"]


def build_research_industry_map(source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """以 screener 的细分行业为研究板块，不将其伪称为 GICS。"""
    required = {"代码", "细分行业", "Nasdaq板块", "数据源", "分类快照UTC"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"基础分类快照缺少字段：{', '.join(sorted(missing))}")
    frame = source.loc[:, ["代码", "细分行业", "Nasdaq板块", "数据源", "分类快照UTC"]].copy()
    frame["代码"] = frame["代码"].astype(str).str.strip().str.upper()
    frame["细分行业"] = frame["细分行业"].fillna("").astype(str).str.strip()
    valid = frame[(frame["代码"] != "") & (frame["细分行业"] != "")].copy()
    valid["行业板块"] = valid["细分行业"]
    valid["Nasdaq细分行业"] = valid["细分行业"]
    conflicts = valid.groupby("代码")["行业板块"].nunique()
    if (conflicts > 1).any():
        samples = conflicts[conflicts > 1].index[:10].tolist()
        raise ValueError(f"同一代码对应多个 Nasdaq 细分行业：{samples}")
    valid = valid.drop_duplicates("代码").loc[:, RESEARCH_COLUMNS].sort_values("代码").reset_index(drop=True)
    excluded = frame[(frame["代码"] == "") | (frame["细分行业"] == "")].loc[:, ["代码", "Nasdaq板块", "细分行业"]].copy()
    excluded["原因"] = "Nasdaq 当前快照没有细分行业，无法构造细粒度研究板块"
    return valid, excluded.sort_values("代码").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="构造板块萌芽统计研究的细粒度 Nasdaq 行业映射")
    parser.add_argument("--source", default="data/nasdaq_10y/industry_map.csv")
    parser.add_argument("--output", default="data/nasdaq_10y/industry_group_map.csv")
    parser.add_argument("--metadata", default="data/nasdaq_10y/industry_group_map_metadata.json")
    parser.add_argument("--unmapped-output", default="data/nasdaq_10y/industry_group_map_unmapped.csv")
    args = parser.parse_args()
    source_path = Path(args.source)
    mapped, unmapped = build_research_industry_map(pd.read_csv(source_path))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(output_path, index=False, encoding="utf-8")
    unmapped.to_csv(Path(args.unmapped_output), index=False, encoding="utf-8")
    metadata = {
        "classification_standard": "Nasdaq screener 细分行业（非 GICS）",
        "source_snapshot": str(source_path),
        "mapped_count": len(mapped),
        "unique_industry_count": int(mapped["行业板块"].nunique()),
        "historical_limit": "当前静态分类快照用于历史研究，存在幸存者偏差和行业归属漂移；研究结论须明确标注偏乐观。",
    }
    Path(args.metadata).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {output_path}：{len(mapped)} 个代码，{metadata['unique_industry_count']} 个细粒度行业。")


if __name__ == "__main__":
    main()
