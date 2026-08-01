"""Resolve configuration resources identically in source and installed builds."""

from __future__ import annotations

from pathlib import Path


def configuration_resource(relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("configuration resource must be a safe relative path")
    package_root = Path(__file__).resolve().parent
    packaged = package_root / "configs" / relative
    if packaged.is_file():
        return packaged
    return package_root.parents[1] / "configs" / relative
