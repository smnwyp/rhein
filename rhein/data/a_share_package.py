"""Build and safely install the external A-share OHLCV data package."""
from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

from .discovery import input_files


PACKAGE_ROOT = "a_share_ohlcv"


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(*, source_dir: Path, destination: Path) -> tuple[int, str]:
    """Create a gzip tarball with one ``a_share_ohlcv/`` top-level directory."""
    files = input_files(source_dir)
    if not files:
        raise ValueError(f"{source_dir}: 没有可打包的 CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{PACKAGE_ROOT}/{path.name}", recursive=False)
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
        if not extracted.is_dir() or not any(extracted.glob("*.csv")):
            raise ValueError("数据包没有可用的 A 股 OHLCV CSV")
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
