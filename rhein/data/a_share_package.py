"""Build and safely install the external A-share OHLCV data package."""
from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .discovery import input_files
from .a_share_industry import (
    GROUP_MANIFEST_FILENAME,
    GROUP_METADATA_FILENAME,
    GROUP_ROOT_NAME,
    SNAPSHOT_FILENAME,
    SNAPSHOT_METADATA_FILENAME,
)


PACKAGE_ROOT = "a_share_ohlcv"
PACKAGE_MARKER = ".rhein_a_share_package_ready"
DEFAULT_A_SHARE_DATA_URL = (
    "https://github.com/smnwyp/rhein/releases/download/"
    "a-share-data-2026-08-12/a_share_ohlcv.tar.gz"
)
DEFAULT_A_SHARE_DATA_SHA256 = "c6dcbd94f5ee4e1d2f95fa57cecedc3dc73e80e50bf1a13f1bcc58278d83118e"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(*, source_dir: Path, destination: Path) -> tuple[int, str]:
    """Create a gzip tarball with OHLCV and optional A-share group metadata."""
    files = input_files(source_dir)
    if not files:
        raise ValueError(f"{source_dir}: 没有可打包的 CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{PACKAGE_ROOT}/{path.name}", recursive=False)
        for filename in (SNAPSHOT_FILENAME, "a_share_industry_map_unmapped.csv", SNAPSHOT_METADATA_FILENAME):
            path = source_dir / filename
            if path.is_file():
                archive.add(path, arcname=f"{PACKAGE_ROOT}/{filename}", recursive=False)
        group_root = source_dir / GROUP_ROOT_NAME
        if group_root.is_dir():
            for folder in sorted(path for path in group_root.iterdir() if path.is_dir()):
                for filename in (GROUP_MANIFEST_FILENAME, GROUP_METADATA_FILENAME):
                    path = folder / filename
                    if path.is_file():
                        archive.add(path, arcname=f"{PACKAGE_ROOT}/{GROUP_ROOT_NAME}/{folder.name}/{filename}", recursive=False)
    return len(files), sha256sum(destination)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("数据包为空")
        for member in members:
            member_path = Path(member.name)
            if member.islnk() or member.issym() or member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"数据包包含不安全路径：{member.name}")
            if not member_path.parts or member_path.parts[0] != PACKAGE_ROOT:
                raise ValueError(f"数据包必须以 {PACKAGE_ROOT}/ 为根目录：{member.name}")
        archive.extractall(destination, filter="data")


def install_archive(*, archive_path: Path, data_root: Path, replace: bool = False) -> Path:
    """Validate and install a package under ``data_root/a_share_ohlcv``."""
    target = data_root / PACKAGE_ROOT
    with tempfile.TemporaryDirectory(dir=data_root.parent, prefix="a_share_install_") as temporary:
        staging = Path(temporary)
        _safe_extract(archive_path, staging)
        extracted = staging / PACKAGE_ROOT
        if not extracted.is_dir() or not input_files(extracted):
            raise ValueError("数据包没有可用的 A 股 OHLCV 数据文件")
        if target.exists():
            if not replace:
                raise FileExistsError(f"{target} 已存在；确认覆盖请传入 --replace")
            shutil.rmtree(target)
        data_root.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(target))
    return target


def download_archive(*, url: str, destination: Path, expected_sha256: str | None = None) -> None:
    """Download a release asset, optionally verifying its SHA-256 digest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    if expected_sha256 and sha256sum(destination).lower() != expected_sha256.lower():
        destination.unlink(missing_ok=True)
        raise ValueError("下载的数据包 SHA-256 校验失败")


def ensure_a_share_data(*, data_root: Path, url: str | None = None, sha256: str | None = None) -> tuple[Path, bool]:
    """Ensure the UI's A-share data directory exists, downloading it only once.

    A deployment may override the public release asset through
    ``A_SHARE_DATA_URL`` and ``A_SHARE_DATA_SHA256`` environment variables or
    Streamlit secrets exposed as environment variables.  The pinned public
    release is the default for the feature/poc deployment.
    """
    target = data_root / PACKAGE_ROOT
    marker = target / PACKAGE_MARKER
    if marker.is_file():
        return target, False
    try:
        if target.is_dir() and input_files(target):
            marker.touch()
            return target, False
    except ValueError:
        # A partial prior install is replaced only after the new package has
        # downloaded and passed its SHA-256 check.
        pass
    package_url = url or os.getenv("A_SHARE_DATA_URL") or DEFAULT_A_SHARE_DATA_URL
    expected_sha256 = sha256 or os.getenv("A_SHARE_DATA_SHA256") or DEFAULT_A_SHARE_DATA_SHA256
    with tempfile.TemporaryDirectory(dir=data_root.parent, prefix="a_share_download_") as temporary:
        archive_path = Path(temporary) / "a_share_ohlcv.tar.gz"
        download_archive(url=package_url, destination=archive_path, expected_sha256=expected_sha256)
        installed = install_archive(archive_path=archive_path, data_root=data_root, replace=target.exists())
    (installed / PACKAGE_MARKER).touch()
    return installed, True
