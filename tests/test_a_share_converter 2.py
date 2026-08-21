from pathlib import Path

from rhein.data.a_share import convert_a_share_file
from rhein.data.ohlc import load_ohlc
from scripts.convert_a_share_ohlcv import main


def test_converts_tongdaxin_a_share_export_to_project_ohlcv(tmp_path: Path) -> None:
    source = tmp_path / "SH#600831.txt"
    source.write_bytes(
        "600831 测试股票 日线 前复权\r\n"
        "      日期\t    开盘\t    最高\t    最低\t    收盘\t    成交量\t    成交额\r\n"
        "06/03/2015\t16.90\t16.96\t15.61\t15.65\t91746209\t1500110464.00\r\n"
        "09/03/2015\t15.49\t16.15\t15.01\t15.87\t52473586\t829215552.00\r\n"
        "#数据来源:通达信\r\n".encode("gb18030")
    )
    output = tmp_path / "SH600831.csv"

    rows = convert_a_share_file(source, output)
    frame = load_ohlc(output)

    assert rows == 2
    assert list(frame.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert frame["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2015-03-06", "2015-03-09"]
    assert frame["Close"].tolist() == [15.65, 15.87]
    assert frame["Volume"].tolist() == [91746209, 52473586]


def test_batch_conversion_continues_after_an_invalid_source(tmp_path: Path, monkeypatch, capsys) -> None:
    valid = tmp_path / "SH#600831.txt"
    valid.write_bytes(
        "600831 测试股票 日线 前复权\r\n"
        "      日期\t    开盘\t    最高\t    最低\t    收盘\t    成交量\t    成交额\r\n"
        "06/03/2015\t16.90\t16.96\t15.61\t15.65\t91746209\t1500110464.00\r\n"
        "#数据来源:通达信\r\n".encode("gb18030")
    )
    invalid = tmp_path / "BJ#920065.txt"
    invalid.write_bytes("920065 空文件\r\n#数据来源:通达信\r\n".encode("gb18030"))
    output_dir = tmp_path / "output"
    monkeypatch.setattr("sys.argv", [
        "convert_a_share_ohlcv.py", "--input-path", str(tmp_path), "--output-dir", str(output_dir),
    ])

    main()

    assert (output_dir / "SH600831.csv").is_file()
    report = output_dir / "conversion_failures.csv"
    assert report.is_file()
    assert "BJ#920065.txt" in report.read_text(encoding="utf-8")
    assert "失败 1 个" in capsys.readouterr().out
