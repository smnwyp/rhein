from pathlib import Path
import shutil

import pytest

from rhein.data import a_share_package
from rhein.data.a_share_package import PACKAGE_MARKER, build_archive, ensure_a_share_data, install_archive, sha256sum


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


def test_ensure_data_downloads_once_and_reuses_a_valid_install(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SH600831.csv").write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")
    archive = tmp_path / "a_share_ohlcv.tar.gz"
    build_archive(source_dir=source, destination=archive)
    calls = []

    def fake_download(*, url: str, destination: Path, expected_sha256: str | None) -> None:
        calls.append((url, expected_sha256))
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(a_share_package, "download_archive", fake_download)
    data_root = tmp_path / "data"

    target, downloaded = ensure_a_share_data(data_root=data_root, url="https://example.test/data.tar.gz", sha256=None)
    reused_target, reused = ensure_a_share_data(data_root=data_root, url="https://example.test/data.tar.gz", sha256=None)

    assert target == reused_target
    assert downloaded is True
    assert reused is False
    assert len(calls) == 1
    assert (target / PACKAGE_MARKER).is_file()


def test_ensure_data_replaces_a_legacy_empty_marker(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SH600831.csv").write_text("Date,Open,High,Low,Close,Volume\n", encoding="utf-8")
    archive = tmp_path / "a_share_ohlcv.tar.gz"
    build_archive(source_dir=source, destination=archive)
    data_root = tmp_path / "data"
    target = install_archive(archive_path=archive, data_root=data_root)
    (target / PACKAGE_MARKER).touch()
    calls = []

    def fake_download(*, url: str, destination: Path, expected_sha256: str | None) -> None:
        calls.append((url, expected_sha256))
        shutil.copyfile(archive, destination)

    monkeypatch.setattr(a_share_package, "download_archive", fake_download)

    updated, downloaded = ensure_a_share_data(data_root=data_root, url="https://example.test/new-data.tar.gz", sha256="expected")

    assert updated == target
    assert downloaded is True
    assert calls == [("https://example.test/new-data.tar.gz", "expected")]
    assert "new-data.tar.gz" in (target / PACKAGE_MARKER).read_text(encoding="utf-8")


def test_package_preserves_industry_snapshot_and_manifest_only_groups(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "SH600831.csv").write_text("Date,Open,High,Low,Close,Volume\n2025-01-02,1,2,1,2,100\n", encoding="utf-8")
    (source / "a_share_industry_map.csv").write_text("symbol,行业板块\nSH600831,银行\n", encoding="utf-8")
    group = source / "industry_groups" / "01_银行"
    group.mkdir(parents=True)
    (group / "group_manifest.csv").write_text("symbol\nSH600831\n", encoding="utf-8")
    (group / "group_metadata.json").write_text('{"industry":"银行","source_data_root":"../.."}\n', encoding="utf-8")
    archive = tmp_path / "a_share_ohlcv.tar.gz"

    build_archive(source_dir=source, destination=archive)
    target = install_archive(archive_path=archive, data_root=tmp_path / "data")

    assert (target / "a_share_industry_map.csv").is_file()
    assert (target / "industry_groups" / "01_银行" / "group_manifest.csv").is_file()
