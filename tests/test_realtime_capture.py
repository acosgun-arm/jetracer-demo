"""Headless browser capture and manifest registration tests."""

from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

import jetracer_sim as sim
from jetracer_sim.realtime_capture import RealTrackCaptureManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPOSITORY_ROOT / "datasets" / "real_track" / "manifest.json"


class FakeVideoWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"video")
        self.frame_count = 0

    @staticmethod
    def isOpened() -> bool:
        return True

    def write(self, _image: np.ndarray) -> None:
        self.frame_count += 1
        with self.path.open("ab") as stream:
            stream.write(b"frame")

    @staticmethod
    def release() -> None:
        pass


class FakeCv2:
    IMWRITE_PNG_COMPRESSION = 16

    @staticmethod
    def imwrite(path: str, image: np.ndarray, _options: list[int]) -> bool:
        Path(path).write_bytes(image.tobytes())
        return True

    @staticmethod
    def VideoWriter_fourcc(*_characters: str) -> int:
        return 1

    @staticmethod
    def VideoWriter(
        path: str,
        _fourcc: int,
        _fps: float,
        _size: tuple[int, int],
    ) -> FakeVideoWriter:
        return FakeVideoWriter(path)


OPTIONS = {
    "image_extension": ".png",
    "png_compression": 1,
    "video_extension": ".mp4",
    "video_fourcc": "mp4v",
    "writer_queue_frames": 4,
    "writer_poll_timeout_s": 0.001,
    "writer_stop_timeout_s": 1.0,
    "snapshot_stop_timeout_s": 1.0,
    "maximum_pending_snapshots": 2,
    "maximum_request_bytes": 1024,
}
METADATA = {
    "split": "calibration",
    "lighting_condition": "daylight_diffuse",
    "track_section": "straight",
    "scene_type": "empty_road",
}


def test_snapshot_and_video_capture_register_raw_media() -> None:
    with TemporaryDirectory(prefix="jetracer-browser-capture-") as directory:
        root = Path(directory)
        manifest = _manifest(root)
        manager = RealTrackCaptureManager(
            FakeCv2(), manifest, "test_4x3_20", OPTIONS
        )
        frame = sim.CapturedFrame(
            frame_id=1,
            image_bgr=np.full((3, 4, 3), 7, dtype=np.uint8),
            captured_at_s=perf_counter(),
        )
        manager.handle_request({"action": "snapshot", **METADATA}, frame)
        manager.handle_request({"action": "start_video", **METADATA}, frame)
        manager.record_frame(frame)
        manager.handle_request({"action": "stop_video"}, frame)
        manager.close()

        saved = loads(manifest.read_text(encoding="utf-8"))
        assert sorted(
            capture["media_type"] for capture in saved["captures"]
        ) == ["image", "video"]
        assert all(
            (root / capture["path"]).is_file()
            for capture in saved["captures"]
        )
        video = next(
            capture
            for capture in saved["captures"]
            if capture["media_type"] == "video"
        )
        assert video["capture_details"] == {
            "target_fps": 20.0,
            "written_frames": 2,
            "dropped_frames": 0,
            "video_fourcc": "mp4v",
        }
        assert manager.status["state"] == "idle"
        assert manager.status["written_frames"] == 2
        assert manager.status["dropped_frames"] == 0


def test_invalid_capture_metadata_is_reported_without_writing() -> None:
    with TemporaryDirectory(prefix="jetracer-browser-capture-invalid-") as directory:
        root = Path(directory)
        manifest = _manifest(root)
        manager = RealTrackCaptureManager(
            FakeCv2(), manifest, "test_4x3_20", OPTIONS
        )
        frame = sim.CapturedFrame(
            frame_id=1,
            image_bgr=np.zeros((3, 4, 3), dtype=np.uint8),
            captured_at_s=perf_counter(),
        )
        manager.handle_request(
            {"action": "snapshot", **METADATA, "split": "unknown"}, frame
        )
        manager.close()
        assert manager.status["state"] == "error"
        assert "invalid capture split" in manager.status["message"]
        assert loads(manifest.read_text(encoding="utf-8"))["captures"] == []


def _manifest(root: Path) -> Path:
    document = loads(TEMPLATE.read_text(encoding="utf-8"))
    document["status"] = "awaiting_capture"
    document["camera_modes"] = [
        {
            "camera_mode_id": "test_4x3_20",
            "camera_id": "test",
            "width": 4,
            "height": 3,
            "fps": 20.0,
            "pixel_format": "bgr",
        }
    ]
    document["captures"] = []
    manifest = root / "manifest.json"
    manifest.write_text(dumps(document), encoding="utf-8")
    return manifest


def main() -> None:
    test_snapshot_and_video_capture_register_raw_media()
    test_invalid_capture_metadata_is_reported_without_writing()


if __name__ == "__main__":
    main()
