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


class FakeLowFpsCapture(FakeCapture):
    def set(self, key: int, value: float) -> bool:
        if key == FakeCV2.CAP_PROP_FPS:
            return False
        return super().set(key, value)


class FakeLowFpsCV2(FakeCV2):
    def VideoCapture(self, *_: object) -> FakeCapture:
        fourcc = self.VideoWriter_fourcc(*"YUY2")
        self.capture = FakeLowFpsCapture(
            [image(0)],
            {
                self.CAP_PROP_FRAME_WIDTH: 1280,
                self.CAP_PROP_FRAME_HEIGHT: 720,
                self.CAP_PROP_FPS: 30.0,
                self.CAP_PROP_FOURCC: fourcc,
            },
        )
        return self.capture


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


def test_avfoundation_identity_gate_rejects_built_in_camera_before_open() -> None:
    original_inventory = frame_source._avfoundation_camera_inventory
    original_import = frame_source._import_cv2
    camera_open_attempted = False

    def forbidden_import() -> Any:
        nonlocal camera_open_attempted
        camera_open_attempted = True
        raise AssertionError("OpenCV must not run when camera identity is absent")

    frame_source._avfoundation_camera_inventory = lambda _: (
        sim.CameraDeviceIdentity(
            "MacBook Pro Camera",
            "MacBook Pro Camera",
            "built-in-1",
        ),
    )
    frame_source._import_cv2 = forbidden_import
    try:
        source = sim.OpenCVCameraFrameSource(
            sim.OpenCVCameraConfig(
                device_index=0,
                width=1280,
                height=720,
                fps=200.0,
                backend="avfoundation",
                identity_requirement=sim.CameraIdentityRequirement(
                    ("Global Shutter Camera",)
                ),
                identity_probe_timeout_s=1.0,
                minimum_resolved_fps_fraction=0.95,
            )
        )
        probe = sim.probe_camera_identity(source.config)
        assert probe.status == "unavailable"
        assert probe.required and not probe.ready
        assert probe.matched_device is None
        assert probe.available_devices[0].name == "MacBook Pro Camera"
        try:
            source.start()
        except sim.FrameSourceError as error:
            assert "required AVFoundation camera is not connected" in str(error)
            assert "MacBook Pro Camera" in str(error)
        else:
            raise AssertionError("built-in camera bypassed the identity gate")
        assert not camera_open_attempted
    finally:
        frame_source._avfoundation_camera_inventory = original_inventory
        frame_source._import_cv2 = original_import


def test_avfoundation_identity_gate_accepts_matching_external_camera() -> None:
    original_inventory = frame_source._avfoundation_camera_inventory
    frame_source._avfoundation_camera_inventory = lambda _: (
        sim.CameraDeviceIdentity(
            "Global Shutter Camera",
            "UVC Camera VendorID_13028 ProductID_21044",
            "external-1",
        ),
    )
    try:
        def exercise() -> None:
            source = sim.OpenCVCameraFrameSource(
                sim.OpenCVCameraConfig(
                    device_index=0,
                    width=1920,
                    height=1200,
                    fps=120.0,
                    backend="avfoundation",
                    identity_requirement=sim.CameraIdentityRequirement(
                        ("Global Shutter Camera",)
                    ),
                    identity_probe_timeout_s=1.0,
                    minimum_resolved_fps_fraction=0.95,
                )
            )
            probe = sim.probe_camera_identity(source.config)
            assert probe.status == "available"
            assert probe.required and probe.ready
            assert probe.matched_device is not None
            assert probe.matched_device.name == "Global Shutter Camera"
            source.start()
            wait_for_publications(source, 1)
            source.stop()

        with_fake_cv2(exercise)
    finally:
        frame_source._avfoundation_camera_inventory = original_inventory


def test_avfoundation_identity_inventory_failure_is_fail_closed() -> None:
    original_inventory = frame_source._avfoundation_camera_inventory
    original_import = frame_source._import_cv2
    camera_open_attempted = False

    def failed_inventory(_: float) -> tuple[sim.CameraDeviceIdentity, ...]:
        raise sim.FrameSourceError("AVFoundation camera inventory failed")

    def forbidden_import() -> Any:
        nonlocal camera_open_attempted
        camera_open_attempted = True
        raise AssertionError("OpenCV must not run after inventory failure")

    frame_source._avfoundation_camera_inventory = failed_inventory
    frame_source._import_cv2 = forbidden_import
    try:
        source = sim.OpenCVCameraFrameSource(
            sim.OpenCVCameraConfig(
                device_index=0,
                width=1280,
                height=720,
                fps=200.0,
                backend="avfoundation",
                identity_requirement=sim.CameraIdentityRequirement(
                    ("Global Shutter Camera",)
                ),
                identity_probe_timeout_s=1.0,
                minimum_resolved_fps_fraction=0.95,
            )
        )
        probe = sim.probe_camera_identity(source.config)
        assert probe.status == "error"
        assert probe.required and not probe.ready
        assert probe.reason == "AVFoundation camera inventory failed"
        try:
            source.start()
        except sim.FrameSourceError as error:
            assert "inventory failed" in str(error)
        else:
            raise AssertionError("camera inventory failure was ignored")
        assert not camera_open_attempted
    finally:
        frame_source._avfoundation_camera_inventory = original_inventory
        frame_source._import_cv2 = original_import


def test_avfoundation_identity_gate_rejects_wrong_negotiated_mode() -> None:
    original_inventory = frame_source._avfoundation_camera_inventory
    original_import = frame_source._import_cv2
    fake_cv2 = FakeLowFpsCV2()
    frame_source._avfoundation_camera_inventory = lambda _: (
        sim.CameraDeviceIdentity(
            "Global Shutter Camera",
            "UVC Camera VendorID_13028 ProductID_21044",
            "external-1",
        ),
    )
    frame_source._import_cv2 = lambda: fake_cv2
    try:
        source = sim.OpenCVCameraFrameSource(
            sim.OpenCVCameraConfig(
                device_index=0,
                width=1280,
                height=720,
                fps=200.0,
                backend="avfoundation",
                identity_requirement=sim.CameraIdentityRequirement(
                    ("Global Shutter Camera",)
                ),
                identity_probe_timeout_s=1.0,
                minimum_resolved_fps_fraction=0.95,
            )
        )
        try:
            source.start()
        except sim.FrameSourceError as error:
            assert "resolved FPS" in str(error)
        else:
            raise AssertionError("wrong camera mode bypassed the acceptance gate")
        assert fake_cv2.capture is not None
        assert not fake_cv2.capture.opened
    finally:
        frame_source._avfoundation_camera_inventory = original_inventory
        frame_source._import_cv2 = original_import


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
    probe = sim.probe_camera_identity(config)
    assert probe.status == "not_required"
    assert not probe.required and probe.ready


def test_camera_rotation_corrects_upside_down_mount() -> None:
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    rotated = frame_source._rotate_image_clockwise(original, 180)
    assert np.array_equal(rotated, original[::-1, ::-1])
    assert rotated.flags.c_contiguous


def main() -> None:
    test_latest_frame_buffer_replaces_stale_frames()
    test_simulator_frame_source_uses_common_contract()
    test_uvc_source_is_latest_only_and_headless()
    test_avfoundation_identity_gate_rejects_built_in_camera_before_open()
    test_avfoundation_identity_gate_accepts_matching_external_camera()
    test_avfoundation_identity_inventory_failure_is_fail_closed()
    test_avfoundation_identity_gate_rejects_wrong_negotiated_mode()
    test_recorded_video_source_uses_latest_frame_contract()
    test_frame_source_module_has_no_gui_calls()
    test_camera_device_path_is_valid_without_opening_it()
    test_camera_rotation_corrects_upside_down_mount()


if __name__ == "__main__":
    main()
