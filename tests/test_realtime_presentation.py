"""Headless tests for real-time demo presentation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np

from jetracer_sim.realtime_presentation import (
    BrowserViewer,
    JsonlTelemetry,
    RollingRate,
    draw_display,
)


class FakeCv2:
    """Record drawing calls without loading a native GUI backend."""

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 1

    def __init__(self) -> None:
        self.text: list[str] = []

    @staticmethod
    def addWeighted(
        image: np.ndarray,
        _image_weight: float,
        _overlay: np.ndarray,
        _overlay_weight: float,
        _offset: float,
    ) -> np.ndarray:
        return image

    @staticmethod
    def rectangle(*_args: object) -> None:
        pass

    @staticmethod
    def circle(*_args: object) -> None:
        pass

    def putText(self, _image: np.ndarray, value: str, *_args: object) -> None:
        self.text.append(value)


def test_rolling_rate_tracks_only_the_requested_window() -> None:
    rate = RollingRate(window_s=1.0)
    for timestamp_s in (0.0, 0.25, 0.5):
        rate.add(timestamp_s)
    assert rate.rate(0.5) == 4.0
    assert rate.rate(1.2) == 4.0
    rate.clear()
    assert rate.rate(1.3) == 0.0


def test_jsonl_telemetry_writes_complete_records() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "telemetry.jsonl"
        with JsonlTelemetry(path, flush_interval_s=1.0) as telemetry:
            telemetry.write({"frame_id": 1, "speed_mps": 0.4})
            telemetry.write({"frame_id": 2, "speed_mps": 0.5})
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert records == [
            {"frame_id": 1, "speed_mps": 0.4},
            {"frame_id": 2, "speed_mps": 0.5},
        ]
        try:
            JsonlTelemetry(path, flush_interval_s=1.0)
        except FileExistsError:
            pass
        else:
            raise AssertionError("telemetry log unexpectedly overwrote a file")


def test_browser_viewer_rejects_invalid_input_before_binding() -> None:
    try:
        BrowserViewer(
            FakeCv2(),
            0,
            viewer_html=b"",
            jpeg_quality=85,
            stream_wait_timeout_s=1.0,
            stop_timeout_s=1.0,
        )
    except ValueError as error:
        assert "HTML" in str(error)
    else:
        raise AssertionError("empty viewer HTML was accepted")


def test_browser_viewer_serves_benchmark_catalog() -> None:
    expected = {"catalog": {"cases": []}, "coverage": {"ready": False}}
    viewer = BrowserViewer(
        FakeCv2(),
        0,
        viewer_html=b"<html></html>",
        jpeg_quality=85,
        stream_wait_timeout_s=1.0,
        stop_timeout_s=1.0,
        benchmark_catalog=expected,
    )
    viewer.start(open_browser=False)
    try:
        with urlopen(f"{viewer.url}/benchmarks", timeout=1.0) as response:
            observed = json.loads(response.read())
        assert observed == expected
        viewer.update_benchmark_catalog({"coverage": {"ready": True}})
        assert viewer.benchmark_catalog_snapshot["coverage"]["ready"]
    finally:
        viewer.stop()


def test_browser_viewer_accepts_capture_requests_and_feed_selection() -> None:
    catalog = {"enabled": True, "camera_mode_id": "test"}
    viewer = BrowserViewer(
        FakeCv2(),
        0,
        viewer_html=b"<html></html>",
        jpeg_quality=85,
        stream_wait_timeout_s=1.0,
        stop_timeout_s=1.0,
        capture_catalog=catalog,
        maximum_capture_request_bytes=1024,
    )
    viewer.start(open_browser=False)
    try:
        payload = {
            "action": "snapshot",
            "split": "calibration",
            "lighting_condition": "daylight_diffuse",
            "track_section": "straight",
            "scene_type": "empty_road",
        }
        request = Request(
            f"{viewer.url}/capture",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=1.0) as response:
            assert response.status == 202
        assert viewer.capture_requests() == [payload]

        feed_request = Request(
            f"{viewer.url}/feed?mode=raw", data=b"", method="POST"
        )
        with urlopen(feed_request, timeout=1.0) as response:
            assert json.loads(response.read())["feed_mode"] == "raw"
        with urlopen(f"{viewer.url}/capture", timeout=1.0) as response:
            status = json.loads(response.read())
        assert status["catalog"] == catalog
        assert status["feed_mode"] == "raw"
    finally:
        viewer.stop()


def test_draw_display_is_independent_of_windowing_backend() -> None:
    cv2 = FakeCv2()
    frame = SimpleNamespace(
        image_bgr=np.zeros((36, 64, 3), dtype=np.uint8)
    )
    camera = SimpleNamespace(fps=200.0)
    vehicle_state = SimpleNamespace(speed_mps=0.55)
    speed = SimpleNamespace(
        perception_age_s=0.012,
        commanded_speed_mps=0.5,
        permitted_speed_mps=0.6,
        certified_speed_limit_mps=0.8,
        certified_speed_limited=False,
        reason="inference_fps",
    )
    model = SimpleNamespace(
        benchmark=None,
        key=1,
        display_name="Test model",
        precision="fp16",
        compression="none",
    )
    statistics = SimpleNamespace(
        replaced_pending_frames=3,
        last_error=None,
    )
    rendered = draw_display(
        cv2,
        frame,
        camera,
        vehicle_state,
        "sim-test",
        None,
        None,
        None,
        speed,
        model,
        statistics,
        None,
        None,
        198.0,
        0.8,
        False,
        True,
    )
    assert rendered.shape == frame.image_bgr.shape
    assert any("speed actual 0.55" in line for line in cv2.text)
    assert any("camera  198.0/200 FPS" in line for line in cv2.text)


def main() -> None:
    test_rolling_rate_tracks_only_the_requested_window()
    test_jsonl_telemetry_writes_complete_records()
    test_browser_viewer_rejects_invalid_input_before_binding()
    test_browser_viewer_serves_benchmark_catalog()
    test_browser_viewer_accepts_capture_requests_and_feed_selection()
    test_draw_display_is_independent_of_windowing_backend()


if __name__ == "__main__":
    main()
