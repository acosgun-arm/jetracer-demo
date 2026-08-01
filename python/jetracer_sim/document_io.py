"""Canonical integrity metadata and crash-safe JSON document persistence."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


INTEGRITY_FIELD = "integrity_sha256"


def document_sha256(document: Mapping[str, Any]) -> str:
    """Hash a JSON object independently of key insertion order."""

    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def with_integrity(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with integrity metadata calculated over the clean body."""

    clean = {key: value for key, value in document.items() if key != INTEGRITY_FIELD}
    return {**clean, INTEGRITY_FIELD: document_sha256(clean)}


def verified_document(document: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Return the clean body and digest, rejecting missing or altered metadata."""

    clean = dict(document)
    digest = clean.pop(INTEGRITY_FIELD, None)
    if not isinstance(digest, str) or digest != document_sha256(clean):
        raise ValueError("document integrity check failed")
    return clean, digest


def atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    """Durably replace a JSON document without exposing a partial target file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
