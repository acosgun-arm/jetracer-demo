"""Tests for deterministic synthetic track clip export."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class FakeVideoSink:
    def __init__(self, path: Path, kind: str, counts: dict[str, int]) -> None:
        self.path = path
        self.kind = kind
        self.counts = counts
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(np.array(frame, copy=True))

    def close(self) -> None:
        self.counts[self.kind] = len(self.frames)
        self.path.write_bytes(f"{self.kind}:{len(self.frames)}".encode())

    def abort(self) -> None:
        self.counts[f"{self.kind}_aborted"] = 1


def main() -> None:
    tool_source = (REPOSITORY_ROOT / "tools/export_synthetic_clip.py").read_text()
    assert 'f"{track_id}-{profile}-{timestamp}"' in tool_source
    with tempfile.TemporaryDirectory(prefix="jetracer-synthetic-clip-") as directory:
        output = Path(directory) / "clip"
        camera = sim.CameraProfile.elp_112()
        camera.width = 160
        camera.height = 100
        camera.fps_numerator = 20
        camera.fps_denominator = 1
        camera.apply_nominal_intrinsics()
        counts: dict[str, int] = {}

        def factory(
            kind: str, path: Path, config: sim.SyntheticClipExportConfig
        ) -> Any:
            assert config.camera.width == 160
            return FakeVideoSink(path, kind, counts)

        summary = sim.export_synthetic_track_clip(
            sim.SyntheticClipExportConfig(
                output_dir=output,
                camera=camera,
                duration_s=0.3,
                cruise_speed_mps=0.5,
            ),
            sink_factory=factory,
        )
        assert summary.frame_count == 6
        assert counts == {"rgb": 6, "semantic": 6}
        assert not (output / "INCOMPLETE").exists()
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["schema_version"] == sim.SYNTHETIC_CLIP_SCHEMA_VERSION
        assert manifest["track"]["id"] == "waveshare_3x2"
        assert manifest["capture"]["frame_count"] == 6
        assert manifest["camera"]["width"] == 160
        assert manifest["files"]["rgb_video"] == "rgb.mp4"
        records = [
            json.loads(line)
            for line in (output / "frames.jsonl").read_text().splitlines()
        ]
        assert len(records) == 6
        assert [record["frame_index"] for record in records] == list(range(6))
        assert records[0]["simulation_time_s"] == 0.0
        assert abs(records[-1]["simulation_time_s"] - 0.25) < 1e-9
        assert len(manifest["sha256"]["rgb.mp4"]) == 64

        try:
            sim.export_synthetic_track_clip(
                sim.SyntheticClipExportConfig(output_dir=output, camera=camera),
                sink_factory=factory,
            )
        except FileExistsError:
            pass
        else:
            raise AssertionError("exporter overwrote an existing clip")


if __name__ == "__main__":
    main()
