"""Tests for latest-only simulator, UVC, and recorded frame sources."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, sleep
from typing import Any

import numpy as np

import jetracer_sim as sim
import jetracer_sim.frame_source as frame_source


def image(value: int) -> np.ndarray:
    return np.full((12, 16, 3), value, dtype=np.uint8)


def captured(frame_id: int) -> sim.CapturedFrame:
    return sim.CapturedFrame(
        frame_id=frame_id,
        image_bgr=image(frame_id),
        captured_at_s=perf_counter(),
    )


def test_latest_frame_buffer_replaces_stale_frames() -> None:
    buffer = sim.LatestFrameBuffer()
    assert buffer.publish(captured(1))
    assert buffer.publish(captured(2))
    newest = buffer.read(0.0)
    assert newest is not None and newest.frame_id == 2
    statistics = buffer.statistics(running=True, failed_reads=0)
    assert statistics.published_frames == 2
    assert statistics.delivered_frames == 1
    assert statistics.replaced_frames == 1


def test_simulator_frame_source_uses_common_contract() -> None:
    scene = sim.Scene.generate(sim.SceneConfig())
    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 64
    camera.height = 36
    camera.apply_nominal_intrinsics()
    source = sim.SimulatorFrameSource(sim.Simulator(scene, camera))
    with source:
        first = source.read(1.0)
        assert first is not None
        assert first.image_bgr.shape == (36, 64, 3)
        assert first.native_frame is not None
        source.set_command(sim.VehicleCommand(0.2, 0.0))
        second = source.read(1.0)
        assert second is not None and second.frame_id > first.frame_id
    assert source.statistics.published_frames >= 2


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], properties: dict[int, float]) -> None:
        self.frames = list(frames)
        self.properties = properties
        self.opened = True
        self.position = 0

    def isOpened(self) -> bool:
        return self.opened

    def set(self, key: int, value: float) -> bool:
        self.properties[key] = value
        if key == FakeCV2.CAP_PROP_POS_FRAMES and value == 0:
            self.position = 0
        return True

    def get(self, key: int) -> float:
        if key == FakeCV2.CAP_PROP_POS_MSEC:
            return self.position * 5.0
        return self.properties.get(key, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        self.opened = False


class FakeCV2:
    CAP_AVFOUNDATION = 1
    CAP_GSTREAMER = 2
    CAP_V4L2 = 3
    CAP_PROP_FRAME_WIDTH = 4
    CAP_PROP_FRAME_HEIGHT = 5
    CAP_PROP_FPS = 6
    CAP_PROP_BUFFERSIZE = 7
    CAP_PROP_FOURCC = 8
    CAP_PROP_POS_MSEC = 9
    CAP_PROP_POS_FRAMES = 10

    def __init__(self) -> None:
        self.capture: FakeCapture | None = None

    def VideoCapture(self, *_: object) -> FakeCapture:
        fourcc = self.VideoWriter_fourcc(*"YUY2")
        self.capture = FakeCapture(
            [image(0), image(1), image(2)],
            {
                self.CAP_PROP_FRAME_WIDTH: 1920,
                self.CAP_PROP_FRAME_HEIGHT: 1200,
                self.CAP_PROP_FPS: 120.0,
                self.CAP_PROP_FOURCC: fourcc,
            },
        )
        return self.capture

    @staticmethod
    def VideoWriter_fourcc(*characters: str) -> int:
        return sum(
            ord(character) << (8 * index)
            for index, character in enumerate(characters)
        )


def wait_for_publications(source: sim.FrameSource, count: int) -> None:
    deadline = perf_counter() + 1.0
    while source.statistics.published_frames < count:
        if perf_counter() >= deadline:
            raise TimeoutError("fake frame source did not publish in time")
        sleep(0.001)


def with_fake_cv2(callback: Any) -> None:
    original = frame_source._import_cv2
    frame_source._import_cv2 = lambda: FakeCV2()
    try:
        callback()
    finally:
        frame_source._import_cv2 = original


def test_uvc_source_is_latest_only_and_headless() -> None:
    def exercise() -> None:
        source = sim.OpenCVCameraFrameSource(
            sim.OpenCVCameraConfig(
                device_index=0,
                width=1920,
                height=1200,
                fps=120.0,
                maximum_consecutive_read_failures=1,
            )
        )
        source.start()
        wait_for_publications(source, 3)
        newest = source.read(1.0)
        assert newest is not None and newest.frame_id == 2
        assert source.resolved_mode == sim.ResolvedCameraMode(
            1920, 1200, 120.0, "YUY2"
        )
        source.stop()
        assert source.statistics.replaced_frames == 2

    with_fake_cv2(exercise)


def test_recorded_video_source_uses_latest_frame_contract() -> None:
    def exercise() -> None:
        with TemporaryDirectory(prefix="jetracer-recorded-source-") as directory:
            path = Path(directory) / "clip.mov"
            path.touch()
            source = sim.RecordedVideoFrameSource(
                sim.RecordedVideoConfig(path, realtime_pacing=False)
            )
            source.start()
            wait_for_publications(source, 3)
            newest = source.read(1.0)
            assert newest is not None and newest.frame_id == 2
            source.stop()
            assert source.statistics.replaced_frames == 2

    with_fake_cv2(exercise)


def test_frame_source_module_has_no_gui_calls() -> None:
    source = Path(frame_source.__file__).read_text(encoding="utf-8")
    for forbidden in ("namedWindow", "imshow", "waitKey", "webbrowser"):
        assert forbidden not in source


def test_camera_device_path_is_valid_without_opening_it() -> None:
    config = sim.OpenCVCameraConfig(
        device_index="/dev/video0",
        width=640,
        height=480,
        fps=30.0,
        backend="v4l2",
    )
    config.validate()


def main() -> None:
    test_latest_frame_buffer_replaces_stale_frames()
    test_simulator_frame_source_uses_common_contract()
    test_uvc_source_is_latest_only_and_headless()
    test_recorded_video_source_uses_latest_frame_contract()
    test_frame_source_module_has_no_gui_calls()
    test_camera_device_path_is_valid_without_opening_it()


if __name__ == "__main__":
    main()
