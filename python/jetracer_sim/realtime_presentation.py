"""Browser UI, display rendering, and telemetry for the real-time demo."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Condition, Thread
from time import perf_counter
from typing import Any, TextIO
from urllib.parse import parse_qs, urlparse
import webbrowser

import numpy as np

import jetracer_sim as sim


__all__ = [
    "BrowserViewer",
    "JsonlTelemetry",
    "RollingRate",
    "detector_result_age_s",
    "draw_display",
    "telemetry_record",
    "unique_log_path",
]


class BrowserViewer:
    """Serve the latest annotated frame and receive keyboard actions."""

    def __init__(
        self,
        cv2: Any,
        port: int,
        *,
        viewer_html: bytes,
        jpeg_quality: int,
        stream_wait_timeout_s: float,
        stop_timeout_s: float,
    ) -> None:
        if not viewer_html:
            raise ValueError("viewer HTML must not be empty")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("browser JPEG quality must be in [1, 100]")
        if stream_wait_timeout_s <= 0.0 or stop_timeout_s <= 0.0:
            raise ValueError("browser timeouts must be positive")
        self._cv2 = cv2
        self._viewer_html = bytes(viewer_html)
        self._jpeg_quality = jpeg_quality
        self._stream_wait_timeout_s = stream_wait_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._condition = Condition()
        self._jpeg: bytes | None = None
        self._version = 0
        self._telemetry: dict[str, Any] = {}
        self._actions: deque[str] = deque()
        self._running = True
        viewer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header(
                        "Content-Length",
                        str(len(viewer._viewer_html)),
                    )
                    self.end_headers()
                    self.wfile.write(viewer._viewer_html)
                elif parsed.path == "/stream.mjpg":
                    self._stream()
                elif parsed.path == "/telemetry":
                    self._send_telemetry()
                else:
                    self.send_error(404)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/action":
                    self.send_error(404)
                    return
                action = parse_qs(parsed.query).get("key", [""])[0]
                viewer._enqueue(action)
                self.send_response(204)
                self.end_headers()

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                version = -1
                try:
                    while True:
                        with viewer._condition:
                            viewer._condition.wait_for(
                                lambda: not viewer._running
                                or viewer._version != version,
                                timeout=viewer._stream_wait_timeout_s,
                            )
                            if not viewer._running:
                                return
                            jpeg = viewer._jpeg
                            version = viewer._version
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    return

            def _send_telemetry(self) -> None:
                snapshot = viewer.telemetry_snapshot
                if not snapshot:
                    self.send_error(503, "telemetry not ready")
                    return
                encoded = json.dumps(
                    snapshot,
                    separators=(",", ":"),
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._server = Server(("127.0.0.1", port), Handler)
        self._thread = Thread(
            target=self._server.serve_forever,
            name="jetracer-viewer",
            daemon=True,
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self, *, open_browser: bool) -> None:
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)

    def publish(self, image: np.ndarray) -> None:
        success, encoded = self._cv2.imencode(
            ".jpg",
            image,
            (self._cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality),
        )
        if not success:
            raise RuntimeError("failed to encode browser viewer frame")
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._version += 1
            self._condition.notify_all()

    def update_telemetry(self, telemetry: dict[str, Any]) -> None:
        with self._condition:
            self._telemetry = telemetry.copy()

    @property
    def telemetry_snapshot(self) -> dict[str, Any]:
        with self._condition:
            return self._telemetry.copy()

    def actions(self) -> list[str]:
        with self._condition:
            actions = list(self._actions)
            self._actions.clear()
            return actions

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=self._stop_timeout_s)

    def _enqueue(self, action: str) -> None:
        if action:
            with self._condition:
                self._actions.append(action)


class RollingRate:
    def __init__(self, window_s: float) -> None:
        if window_s <= 0.0:
            raise ValueError("rate window must be positive")
        self.window_s = window_s
        self._events: deque[float] = deque()

    def add(self, timestamp_s: float) -> None:
        self._events.append(timestamp_s)
        self._trim(timestamp_s)

    def clear(self) -> None:
        self._events.clear()

    def rate(self, now_s: float) -> float:
        self._trim(now_s)
        if len(self._events) < 2:
            return 0.0
        duration_s = self._events[-1] - self._events[0]
        return (len(self._events) - 1) / max(duration_s, 1e-9)

    def _trim(self, now_s: float) -> None:
        cutoff_s = now_s - self.window_s
        while self._events and self._events[0] < cutoff_s:
            self._events.popleft()


class JsonlTelemetry:
    def __init__(self, path: Path, flush_interval_s: float) -> None:
        if flush_interval_s <= 0.0:
            raise ValueError("telemetry flush interval must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file: TextIO = path.open("x", encoding="utf-8")
        self._last_flush_s = perf_counter()
        self._flush_interval_s = flush_interval_s

    def write(self, record: dict[str, Any]) -> None:
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        now_s = perf_counter()
        if now_s - self._last_flush_s >= self._flush_interval_s:
            self._file.flush()
            self._last_flush_s = now_s

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> JsonlTelemetry:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def detector_result_age_s(
    result: sim.TimedDetections | None,
    now_s: float,
) -> float | None:
    if result is None:
        return None
    return result.metrics.end_to_end_latency_s + max(
        0.0,
        now_s - result.metrics.completed_at_s,
    )


def unique_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path("build/telemetry") / f"realtime-{timestamp}"
    candidate = base.with_suffix(".jsonl")
    index = 1
    while candidate.exists():
        candidate = Path(f"{base}-{index}.jsonl")
        index += 1
    return candidate


def draw_display(
    cv2: Any,
    frame: sim.CapturedFrame,
    camera: sim.CameraProfile,
    vehicle_state: sim.VehicleStateSample,
    platform_id: str,
    latest: sim.TimedSegmentation | None,
    latest_detections: sim.TimedDetections | None,
    steering: sim.SteeringDecision | None,
    speed: sim.GovernorDecision,
    active_model: sim.ModelVariant,
    statistics: sim.InferenceWorkerStatistics,
    detection_statistics: sim.InferenceWorkerStatistics | None,
    stop_decision: sim.StopSignDecision | None,
    source_fps: float,
    requested_speed_mps: float,
    paused: bool,
    show_labels: bool,
) -> np.ndarray:
    image = np.array(frame.image_bgr, copy=True)
    if show_labels and latest is not None:
        overlay = np.zeros_like(image)
        road = latest.prediction.labels == 1
        lane = latest.prediction.labels == 2
        overlay[road] = (70, 130, 70)
        overlay[lane] = (40, 210, 240)
        image = cv2.addWeighted(image, 0.72, overlay, 0.28, 0.0)

    if show_labels and latest_detections is not None:
        for detection in latest_detections.detections:
            x0, y0, x1, y1 = (
                int(round(value)) for value in detection.bbox_xyxy
            )
            cv2.rectangle(image, (x0, y0), (x1, y1), (80, 170, 255), 2)
            label = detection.label or str(detection.class_id)
            label += f" {detection.confidence:.2f}"
            if detection.range_m is not None:
                label += f" {detection.range_m:.2f} m"
            cv2.putText(
                image,
                label,
                (x0, max(16, y0 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (80, 170, 255),
                1,
                cv2.LINE_AA,
            )

    if steering is not None and steering.target_pixel_xy is not None:
        target = tuple(int(round(value)) for value in steering.target_pixel_xy)
        cv2.circle(image, target, 7, (30, 230, 255), 2, cv2.LINE_AA)

    metrics = latest.metrics if latest is not None else None
    inference_fps = metrics.effective_fps if metrics is not None else 0.0
    inference_ms = metrics.ewma_latency_s * 1000.0 if metrics is not None else 0.0
    age_ms = speed.perception_age_s * 1000.0
    if not np.isfinite(age_ms):
        age_text = "waiting"
    else:
        age_text = f"{age_ms:.1f} ms"
    benchmark_text = (
        "UNBENCHMARKED"
        if active_model.benchmark is None
        else f"BENCH {active_model.benchmark.measured_fps:.1f} FPS"
    )
    actual_speed_text = (
        "unavailable"
        if vehicle_state.speed_mps is None
        else f"{vehicle_state.speed_mps:.2f}"
    )
    lines = [
        f"{platform_id}  [{active_model.key}] {active_model.display_name}  "
        f"{benchmark_text}  "
        f"{active_model.precision}/{active_model.compression}",
        f"camera {source_fps:6.1f}/{camera.fps:.0f} FPS   "
        f"inference {inference_fps:6.1f} FPS  {inference_ms:5.1f} ms   "
        f"result age {age_text}   dropped {statistics.replaced_pending_frames}",
        f"speed actual {actual_speed_text}  command "
        f"{speed.commanded_speed_mps:.2f}  permitted "
        f"{speed.permitted_speed_mps:.2f}  requested "
        f"{requested_speed_mps:.2f} m/s   limit {speed.reason}",
    ]
    if detection_statistics is not None:
        detection_metrics = (
            latest_detections.metrics if latest_detections is not None else None
        )
        detection_fps = (
            detection_metrics.effective_fps
            if detection_metrics is not None
            else 0.0
        )
        detection_ms = (
            detection_metrics.ewma_latency_s * 1000.0
            if detection_metrics is not None
            else 0.0
        )
        stop_state = (
            stop_decision.state.value if stop_decision is not None else "waiting"
        )
        detection_count = (
            len(latest_detections.detections)
            if latest_detections is not None
            else 0
        )
        lines.append(
            f"detector {detection_fps:6.1f} FPS  {detection_ms:5.1f} ms   "
            f"objects {detection_count}   dropped "
            f"{detection_statistics.replaced_pending_frames}   stop {stop_state}"
        )
    lines.append(
        "1-9 model  [/] requested speed  P pause  L labels  "
        "R control reset  SPACE stop  Q quit"
    )
    line_height = 27
    hud_height = 10 + len(lines) * line_height
    cv2.rectangle(
        image,
        (0, 0),
        (image.shape[1], hud_height),
        (20, 20, 20),
        -1,
    )
    for index, line in enumerate(lines):
        colour = (245, 245, 245) if index < 3 else (200, 200, 200)
        cv2.putText(
            image,
            line,
            (14, 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56 if index < 3 else 0.50,
            colour,
            1,
            cv2.LINE_AA,
        )
    if paused:
        cv2.putText(
            image,
            "PAUSED",
            (image.shape[1] - 125, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 210, 255),
            2,
            cv2.LINE_AA,
        )
    if statistics.last_error is not None:
        cv2.putText(
            image,
            statistics.last_error,
            (14, image.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (40, 40, 240),
            1,
            cv2.LINE_AA,
        )
    if detection_statistics is not None and detection_statistics.last_error:
        cv2.putText(
            image,
            detection_statistics.last_error,
            (14, image.shape[0] - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (40, 40, 240),
            1,
            cv2.LINE_AA,
        )
    return image


def telemetry_record(
    now_s: float,
    started_at_s: float,
    frame: sim.CapturedFrame,
    camera: sim.CameraProfile,
    vehicle_state: sim.VehicleStateSample,
    platform: sim.PlatformConfiguration,
    latest: sim.TimedSegmentation | None,
    steering: sim.SteeringDecision | None,
    speed: sim.GovernorDecision,
    statistics: sim.InferenceWorkerStatistics,
    source_fps: float,
    requested_speed_mps: float,
    active_model: sim.ModelVariant,
    available_models: tuple[sim.ModelVariant, ...],
    paused: bool,
    show_labels: bool,
    latest_detections: sim.TimedDetections | None = None,
    detection_statistics: sim.InferenceWorkerStatistics | None = None,
    active_detector: sim.DetectionModelVariant | None = None,
    stop_decision: sim.StopSignDecision | None = None,
    frame_source_statistics: sim.FrameSourceStatistics | None = None,
    actuator_status: sim.VehicleActuatorStatus | None = None,
    include_model_catalog: bool = False,
    system_health: sim.SystemHealthSnapshot | None = None,
) -> dict[str, Any]:
    metrics = latest.metrics if latest is not None else None
    detection_metrics = (
        latest_detections.metrics if latest_detections is not None else None
    )
    detection_age_s = detector_result_age_s(latest_detections, now_s)
    benchmark = active_model.benchmark
    record = {
        "wall_time_s": now_s - started_at_s,
        "platform_id": platform.platform_id,
        "platform_mode": platform.mode,
        "simulation_time_s": (
            frame.source_timestamp_s if platform.mode == "sim" else None
        ),
        "source_timestamp_s": frame.source_timestamp_s,
        "captured_at_s": frame.captured_at_s,
        "source_frame_id": frame.frame_id,
        "result_frame_id": metrics.frame_id if metrics is not None else None,
        "active_model_id": active_model.model_id,
        "active_model_key": active_model.key,
        "active_model_name": active_model.display_name,
        "model_backend": active_model.backend,
        "model_precision": active_model.precision,
        "model_compression": active_model.compression,
        "benchmark_fps": active_model.benchmark_fps,
        "benchmark_source": benchmark.source if benchmark is not None else None,
        "benchmark_p99_latency_s": (
            benchmark.p99_latency_s if benchmark is not None else None
        ),
        "benchmark_environment": (
            benchmark.environment if benchmark is not None else None
        ),
        "benchmark_details": (
            benchmark.details if benchmark is not None else None
        ),
        "result_model_id": speed.model_id,
        "model_generation": (
            metrics.model_generation if metrics is not None else None
        ),
        "camera_fps": source_fps,
        "measured_camera_fps": source_fps,
        "camera_target_fps": camera.fps,
        "effective_inference_fps": speed.effective_fps,
        "inference_latency_s": (
            metrics.inference_latency_s if metrics is not None else None
        ),
        "ewma_inference_latency_s": (
            metrics.ewma_latency_s if metrics is not None else None
        ),
        "perception_age_s": (
            speed.perception_age_s
            if np.isfinite(speed.perception_age_s)
            else None
        ),
        "requested_speed_mps": requested_speed_mps,
        "governor_target_speed_mps": speed.target_speed_mps,
        "permitted_speed_mps": speed.permitted_speed_mps,
        "commanded_speed_mps": speed.commanded_speed_mps,
        "actual_speed_mps": vehicle_state.speed_mps,
        "actual_steering_rad": vehicle_state.steering_rad,
        "vehicle_state_source": vehicle_state.source,
        "vehicle_state_quality": vehicle_state.quality,
        "vehicle_state_sequence_id": vehicle_state.sequence_id,
        "vehicle_state_confidence": vehicle_state.confidence,
        "vehicle_state_age_s": vehicle_state.age_s(now_s),
        "speed_limit_reason": speed.reason,
        "steering_rad": steering.steering_rad if steering is not None else 0.0,
        "tracking_confidence": (
            steering.confidence if steering is not None else 0.0
        ),
        "submitted_frames": statistics.submitted_frames,
        "completed_frames": statistics.completed_frames,
        "replaced_pending_frames": statistics.replaced_pending_frames,
        "discarded_results": statistics.discarded_results,
        "failed_frames": statistics.failed_frames,
        "detector_model_id": (
            active_detector.model_id if active_detector is not None else None
        ),
        "detector_model_name": (
            active_detector.display_name if active_detector is not None else None
        ),
        "detector_effective_fps": (
            detection_metrics.effective_fps
            if detection_metrics is not None
            else None
        ),
        "detector_latency_s": (
            detection_metrics.ewma_latency_s
            if detection_metrics is not None
            else None
        ),
        "detector_age_s": detection_age_s,
        "detected_object_count": (
            len(latest_detections.detections)
            if latest_detections is not None
            else 0
        ),
        "detector_replaced_pending_frames": (
            detection_statistics.replaced_pending_frames
            if detection_statistics is not None
            else 0
        ),
        "detector_failed_frames": (
            detection_statistics.failed_frames
            if detection_statistics is not None
            else 0
        ),
        "detector_error": (
            detection_statistics.last_error
            if detection_statistics is not None
            else None
        ),
        "stop_state": (
            stop_decision.state.value if stop_decision is not None else None
        ),
        "stop_reason": (
            stop_decision.reason if stop_decision is not None else None
        ),
        "stop_range_m": (
            stop_decision.nearest_range_m if stop_decision is not None else None
        ),
        "stop_speed_limit_mps": (
            stop_decision.speed_limit_mps if stop_decision is not None else None
        ),
        "paused": paused,
        "show_labels": show_labels,
        "capture_published_frames": (
            frame_source_statistics.published_frames
            if frame_source_statistics is not None
            else None
        ),
        "capture_replaced_frames": (
            frame_source_statistics.replaced_frames
            if frame_source_statistics is not None
            else None
        ),
        "capture_failed_reads": (
            frame_source_statistics.failed_reads
            if frame_source_statistics is not None
            else None
        ),
        "actuator_driver": (
            actuator_status.driver if actuator_status is not None else None
        ),
        "actuator_output_enabled": (
            actuator_status.output_enabled
            if actuator_status is not None
            else None
        ),
        "actuator_watchdog_armed": (
            actuator_status.watchdog_armed
            if actuator_status is not None
            else None
        ),
        "actuator_watchdog_expirations": (
            actuator_status.watchdog_expirations
            if actuator_status is not None
            else None
        ),
        "actuator_emergency_stop_reason": (
            actuator_status.emergency_stop_reason
            if actuator_status is not None
            else None
        ),
        "actuator_command_age_s": (
            max(now_s - actuator_status.last_command_at_s, 0.0)
            if actuator_status is not None
            and actuator_status.last_command_at_s is not None
            else None
        ),
        "system_health_age_s": (
            system_health.age_s(now_s) if system_health is not None else None
        ),
        "maximum_temperature_c": (
            system_health.maximum_temperature_c
            if system_health is not None
            else None
        ),
        "temperature_sensor_count": (
            system_health.temperature_sensor_count
            if system_health is not None
            else 0
        ),
    }
    if include_model_catalog:
        record["available_models"] = [
            {
                "key": model.key,
                "model_id": model.model_id,
                "display_name": model.display_name,
                "benchmark_fps": model.benchmark_fps,
                "benchmark_source": (
                    model.benchmark.source
                    if model.benchmark is not None
                    else None
                ),
            }
            for model in available_models
        ]
    return record
