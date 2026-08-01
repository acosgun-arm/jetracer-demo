"""Canonical integrity and atomic JSON persistence regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jetracer_sim.document_io import (
    atomic_write_json,
    document_sha256,
    verified_document,
    with_integrity,
)


def test_integrity_is_order_independent_and_non_mutating() -> None:
    first = {"beta": [1, 2], "alpha": {"ready": True}}
    second = {"alpha": {"ready": True}, "beta": [1, 2]}
    assert document_sha256(first) == document_sha256(second)
    protected = with_integrity(first)
    assert "integrity_sha256" not in first
    clean, digest = verified_document(protected)
    assert clean == first
    assert digest == document_sha256(first)
    protected["alpha"]["ready"] = False
    try:
        verified_document(protected)
    except ValueError as error:
        assert "integrity" in str(error)
    else:
        raise AssertionError("altered document passed integrity verification")


def test_atomic_write_replaces_complete_document() -> None:
    with TemporaryDirectory(prefix="jetracer-document-test-") as directory:
        path = Path(directory) / "nested/state.json"
        atomic_write_json(path, {"generation": 1})
        atomic_write_json(path, {"generation": 2, "ready": True})
        assert json.loads(path.read_text(encoding="utf-8")) == {
            "generation": 2,
            "ready": True,
        }
        assert not list(path.parent.glob(f".{path.name}.tmp-*"))


def test_failed_replace_preserves_target_and_cleans_temporary() -> None:
    with TemporaryDirectory(prefix="jetracer-document-test-") as directory:
        path = Path(directory) / "state.json"
        atomic_write_json(path, {"generation": 1})
        original = path.read_bytes()
        try:
            with patch(
                "jetracer_sim.document_io.os.replace",
                side_effect=OSError("injected replacement failure"),
            ):
                atomic_write_json(path, {"generation": 2})
        except OSError as error:
            assert "injected" in str(error)
        else:
            raise AssertionError("injected atomic-write failure was ignored")
        assert path.read_bytes() == original
        assert not list(path.parent.glob(f".{path.name}.tmp-*"))


def main() -> None:
    test_integrity_is_order_independent_and_non_mutating()
    test_atomic_write_replaces_complete_document()
    test_failed_replace_preserves_target_and_cleans_temporary()


if __name__ == "__main__":
    main()
