from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rhein.data.a_share_industry import (
    AShareIndustryError,
    EASTMONEY_A_SHARE_FILTER,
    GROUP_MANIFEST_FILENAME,
    GROUP_METADATA_FILENAME,
    GROUP_ROOT_NAME,
    LEGULEGU_SW_COMPOSITION_URL,
    LEGULEGU_SW_OVERVIEW_URL,
    build_industry_snapshot,
    fetch_eastmoney_a_share_snapshot,
    fetch_legulegu_sw_level_one_snapshot,
    market_prefixed_symbol,
    write_industry_groups,
)
from rhein.data.discovery import input_files
from rhein.ui.presets import available_data_scopes
import rhein.ui.presets as preset_ui


def test_eastmoney_codes_normalize_to_the_local_market_prefixed_convention() -> None:
    assert market_prefixed_symbol("600000", 1) == "SH600000"
    assert market_prefixed_symbol("300750", 0) == "SZ300750"
    assert market_prefixed_symbol("920363", 0) == "BJ920363"
    assert market_prefixed_symbol("not-a-code", 0) is None


def test_industry_snapshot_only_classifies_the_installed_ohlcv_universe() -> None:
    source = pd.DataFrame([
        {"symbol": "SH600000", "代码": "600000", "名称": "浦发银行", "行业板块": "银行"},
        {"symbol": "SZ300750", "代码": "300750", "名称": "宁德时代", "行业板块": "电池"},
        {"symbol": "SZ000001", "代码": "000001", "名称": "平安银行", "行业板块": ""},
    ])

    mapped, unmapped = build_industry_snapshot(
        ["SH600000", "SZ300750", "SZ000001", "BAD"], source, snapshot_at="2026-08-31T00:00:00+00:00",
    )

    assert mapped[["symbol", "行业板块"]].to_dict("records") == [
        {"symbol": "SZ300750", "行业板块": "电池"},
        {"symbol": "SH600000", "行业板块": "银行"},
    ]
    assert unmapped.to_dict("records") == [{"symbol": "SZ000001", "原因": "当前行业快照未包含该代码或行业字段为空"}]
    assert set(mapped["数据源"]) == {"申万 2021 一级行业（乐咕乐股当前成分股快照）"}


def test_snapshot_fetch_pages_through_the_a_share_list_and_requires_the_industry_field() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def __init__(self, page: int) -> None:
            self.page = page

        def json(self) -> dict:
            code = "600000" if self.page == 1 else "300750"
            market = 1 if self.page == 1 else 0
            name = "浦发银行" if self.page == 1 else "宁德时代"
            industry = "银行" if self.page == 1 else "电池"
            return {"data": {"total": 101, "diff": [{"f12": code, "f13": market, "f14": name, "f100": industry}]}}

    class Session:
        def __init__(self) -> None:
            self.kwargs: dict | None = None

        def get(self, _url: str, **kwargs: object) -> Response:
            self.kwargs = kwargs
            return Response(int(kwargs["params"]["pn"]))

    session = Session()
    frame = fetch_eastmoney_a_share_snapshot(session=session)  # type: ignore[arg-type]

    assert session.kwargs is not None
    assert session.kwargs["params"]["fs"] == EASTMONEY_A_SHARE_FILTER
    assert "f100" in session.kwargs["params"]["fields"]
    assert len(frame) == 2
    assert set(frame["symbol"]) == {"SH600000", "SZ300750"}


def test_snapshot_fetch_rejects_an_unexpected_api_shape() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"data": {"diff": [{"f12": "600000"}]}}

    class Session:
        def get(self, _url: str, **_kwargs: object) -> Response:
            return Response()

    with pytest.raises(AShareIndustryError, match="缺少字段"):
        fetch_eastmoney_a_share_snapshot(session=Session())  # type: ignore[arg-type]


def test_sw_level_one_snapshot_uses_each_public_constituent_page_once_and_can_resume_from_cache(tmp_path: Path) -> None:
    overview = '''<li class="level1Item"><div id="801010.SI"></div><div class="lg-industries-item-number">农林牧渔(2)</div></li>'''
    composition = '''<tr class="index-basic-composition-item" data-stock-code="000019.SZ"></tr>
    <tr class="index-basic-composition-item" data-stock-code="600127.SH"></tr>'''

    class Response:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class Session:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def get(self, url: str, **kwargs: object) -> Response:
            self.calls.append((url, kwargs.get("params")))
            return Response(overview if url == LEGULEGU_SW_OVERVIEW_URL else composition)

    session = Session()
    frame = fetch_legulegu_sw_level_one_snapshot(session=session, cache_dir=tmp_path)  # type: ignore[arg-type]

    assert frame.to_dict("records") == [
        {"symbol": "SH600127", "代码": "600127", "名称": "", "行业板块": "农林牧渔"},
        {"symbol": "SZ000019", "代码": "000019", "名称": "", "行业板块": "农林牧渔"},
    ]
    assert session.calls == [
        (LEGULEGU_SW_OVERVIEW_URL, None),
        (LEGULEGU_SW_COMPOSITION_URL, {"industryCode": "801010.SI"}),
    ]
    cached = fetch_legulegu_sw_level_one_snapshot(session=Session(), cache_dir=tmp_path)  # type: ignore[arg-type]
    assert cached.equals(frame)


def test_industry_group_manifest_loads_ohlcv_from_its_shared_source_root(tmp_path: Path) -> None:
    data_root = tmp_path / "a_share_ohlcv"
    data_root.mkdir()
    (data_root / "SH600000.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n2025-01-02,1,2,1,2,100\n", encoding="utf-8",
    )
    mapped = pd.DataFrame([{
        "symbol": "SH600000", "代码": "600000", "名称": "浦发银行", "行业板块": "银行",
        "数据源": "测试", "分类快照UTC": "2026-08-31T00:00:00+00:00",
    }])
    groups = write_industry_groups(
        data_root=data_root, mapped=mapped, unmapped=pd.DataFrame(columns=["symbol", "原因"]),
        snapshot_at="2026-08-31T00:00:00+00:00",
    )

    assert input_files(groups[0]) == [data_root / "SH600000.csv"]
    assert (groups[0] / GROUP_MANIFEST_FILENAME).is_file()
    assert json.loads((groups[0] / GROUP_METADATA_FILENAME).read_text(encoding="utf-8"))["source_data_root"] == "../.."
    saved_combos = groups[0] / "saved_combos.json"
    saved_combos.write_text('{"combos": []}\n', encoding="utf-8")
    write_industry_groups(
        data_root=data_root, mapped=mapped, unmapped=pd.DataFrame(columns=["symbol", "原因"]),
        snapshot_at="2026-09-01T00:00:00+00:00",
    )
    assert saved_combos.is_file()


def test_available_data_scopes_exposes_written_industry_groups(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a_share_root = tmp_path / "a_share_ohlcv"
    a_share_root.mkdir()
    (a_share_root / "SH600000.parquet").write_bytes(b"placeholder")
    group = a_share_root / GROUP_ROOT_NAME / "01_银行"
    group.mkdir(parents=True)
    (group / GROUP_MANIFEST_FILENAME).write_text("symbol\nSH600000\n", encoding="utf-8")
    (group / GROUP_METADATA_FILENAME).write_text(
        json.dumps({"industry": "银行", "source_data_root": "../..", "classification_source": "申万 2021 一级行业（乐咕乐股当前成分股快照）"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(preset_ui, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preset_ui, "NASDAQ_ROOT", tmp_path / "nasdaq_10y")
    monkeypatch.setattr(preset_ui, "GROUP_ROOT", tmp_path / "nasdaq_10y" / "groups")
    monkeypatch.setattr(preset_ui, "A_SHARE_ROOT", a_share_root)

    scopes = available_data_scopes()

    assert scopes["A 股行业 · 银行（1 个标的）"] == str(group)


def test_a_share_snapshot_csvs_do_not_inflate_the_universe_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    a_share_root = tmp_path / "a_share_ohlcv"
    a_share_root.mkdir()
    (a_share_root / "SH600000.parquet").write_bytes(b"placeholder")
    (a_share_root / "a_share_industry_map.csv").write_text("symbol\nSH600000\n", encoding="utf-8")
    (a_share_root / "a_share_industry_map_unmapped.csv").write_text("symbol\nSZ000001\n", encoding="utf-8")
    monkeypatch.setattr(preset_ui, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(preset_ui, "NASDAQ_ROOT", tmp_path / "nasdaq_10y")
    monkeypatch.setattr(preset_ui, "GROUP_ROOT", tmp_path / "nasdaq_10y" / "groups")
    monkeypatch.setattr(preset_ui, "A_SHARE_ROOT", a_share_root)

    assert "全部 A 股（1 个可回测标的，前复权日线）" in available_data_scopes()
