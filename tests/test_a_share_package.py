from pathlib import Path

import pytest

from rhein.data.a_share_package import build_archive, install_archive, sha256sum


def test_package_round_trip_installs_csvs_under_the_expected_data_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SH600831.csv").write_text(
        "Date,Open,High,Low,Close,Volume\n2025-01-02,1,2,1,2,100\n", encoding="utf-8"
    )
    archive = tmp_path / "a_share_ohlcv.tar.gz"

    count, digest = build_archive(source_dir=source, destination=archive)
    target = install_archive(archive_path=archive, data_root=tmp_path / "data")

    assert count == 1
    assert digest == sha256sum(archive)
    assert (target / "SH600831.csv").is_file()


def test_package_install_refuses_to_overwrite_data_without_explicit_consent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SH600831.csv").write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")
    archive = tmp_path / "a_share_ohlcv.tar.gz"
    build_archive(source_dir=source, destination=archive)
    data_root = tmp_path / "data"
    install_archive(archive_path=archive, data_root=data_root)

    with pytest.raises(FileExistsError, match="--replace"):
        install_archive(archive_path=archive, data_root=data_root)
