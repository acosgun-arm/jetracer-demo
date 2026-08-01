"""Headless frame sources with latest-frame-only delivery semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from threading import Condition, Event, Lock, Thread
from time import perf_counter
from typing import Any

import numpy as np

from ._native import Simulator, VehicleCommand
from .configuration import runtime_config_section


_DEFAULTS = runtime_config_section("frame_source")
_LIVE_DEFAULTS = _DEFAULTS["live_camera"]
_RECORDED_DEFAULTS = _DEFAULTS["recorded_video"]


class FrameSourceError(RuntimeError):
    """Raised when a frame producer fails or cannot be opened."""


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """One image timestamped at capture delivery, not inference submission."""

    frame_id: int
    image_bgr: np.ndarray
    captured_at_s: float
    source_timestamp_s: float | None = None
    native_frame: Any | None = None

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame ID must not be negative")
        if (
            not isinstance(self.image_bgr, np.ndarray)
            or self.image_bgr.dtype != np.uint8
            or self.image_bgr.ndim != 3
            or self.image_bgr.shape[2] != 3
        ):
            raise ValueError("captured image must be an HxWx3 uint8 array")
        if not isfinite(self.captured_at_s):
            raise ValueError("capture timestamp must be finite")
        if self.source_timestamp_s is not None and not isfinite(
            self.source_timestamp_s
        ):
            raise ValueError("source timestamp must be finite when provided")


@dataclass(frozen=True, slots=True)
class FrameSourceStatistics:
    published_frames: int
    delivered_frames: int
    replaced_frames: int
    read_timeouts: int
    failed_reads: int
    running: bool
    pending: bool
    last_error: str | None


class LatestFrameBuffer:
    """A one-frame mailbox that never lets capture latency accumulate."""

    def __init__(self) -> None:
        self._condition = Condition()
        self._latest: CapturedFrame | None = None
        self._closed = False
        self._last_error: str | None = None
        self._published_frames = 0
        self._delivered_frames = 0
        self._replaced_frames = 0
        self._read_timeouts = 0

    def publish(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._published_frames += 1
            if self._latest is not None:
                self._replaced_frames += 1
            self._latest = frame
            self._condition.notify_all()
            return True

    def read(
        self,
        timeout_s: float = float(_DEFAULTS["read_timeout_s"]),
    ) -> CapturedFrame | None:
        if timeout_s < 0.0:
            raise ValueError("frame read timeout must not be negative")
        deadline = perf_counter() + timeout_s
        with self._condition:
            while self._latest is None and not self._closed:
                remaining_s = deadline - perf_counter()
                if remaining_s <= 0.0:
                    self._read_timeouts += 1
                    return None
                self._condition.wait(remaining_s)
            if self._latest is not None:
                frame = self._latest
                self._latest = None
                self._delivered_frames += 1
                return frame
            if self._last_error is not None:
                raise FrameSourceError(self._last_error)
            return None

    def close(self, error: str | None = None) -> None:
        with self._condition:
            self._closed = True
            self._last_error = error
            self._condition.notify_all()

    def statistics(
        self,
        *,
        running: bool,
        failed_reads: int,
    ) -> FrameSourceStatistics:
        with self._condition:
            return FrameSourceStatistics(
                published_frames=self._published_frames,
                delivered_frames=self._delivered_frames,
                replaced_frames=self._replaced_frames,
                read_timeouts=self._read_timeouts,
                failed_reads=failed_reads,
                running=running,
                pending=self._latest is not None,
                last_error=self._last_error,
            )


class FrameSource(ABC):
    @abstractmethod
    def start(self) -> None:
        """Open the producer and begin capture."""

    @abstractmethod
    def read(self, timeout_s: float | None = None) -> CapturedFrame | None:
        """Return the newest frame, or None on timeout/end of stream."""

    @abstractmethod
    def stop(self, timeout_s: float | None = None) -> None:
        """Stop capture and release its resources."""

    @property
    @abstractmethod
    def statistics(self) -> FrameSourceStatistics:
        """Return a thread-safe statistics snapshot."""

    def __enter__(self) -> FrameSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


class _ThreadedFrameSource(FrameSource):
    def __init__(self) -> None:
        self._buffer = LatestFrameBuffer()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._started = False
        self._failed_reads = 0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("frame sources cannot be restarted")
        self._prepare()
        self._started = True
        self._thread = Thread(
            target=self._run,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def read(self, timeout_s: float | None = None) -> CapturedFrame | None:
        resolved_timeout = (
            float(_DEFAULTS["read_timeout_s"])
            if timeout_s is None
            else timeout_s
        )
        return self._buffer.read(resolved_timeout)

    def stop(self, timeout_s: float | None = None) -> None:
        resolved_timeout = (
            float(_DEFAULTS["stop_timeout_s"])
            if timeout_s is None
            else timeout_s
        )
        if resolved_timeout <= 0.0:
            raise ValueError("frame source stop timeout must be positive")
        self._stop_event.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(resolved_timeout)
        if thread.is_alive():
            self._interrupt()
            thread.join(resolved_timeout)
        if thread.is_alive():
            raise TimeoutError("frame source did not stop")
        self._thread = None

    @property
    def statistics(self) -> FrameSourceStatistics:
        thread = self._thread
        return self._buffer.statistics(
            running=thread is not None and thread.is_alive(),
            failed_reads=self._failed_reads,
        )

    @property
    @abstractmethod
    def _thread_name(self) -> str:
        pass

    def _prepare(self) -> None:
        pass

    def _interrupt(self) -> None:
        pass

    def _release(self) -> None:
        pass

    @abstractmethod
    def _next_frame(self) -> CapturedFrame | None:
        pass

    def _run(self) -> None:
        error_message: str | None = None
        try:
            while not self._stop_event.is_set():
                frame = self._next_frame()
                if frame is None:
                    break
                if not self._buffer.publish(frame):
                    break
        except Exception as error:  # Keep control processes diagnosable.
            error_message = f"{type(error).__name__}: {error}"
        finally:
            try:
                self._release()
            except Exception as error:
                if error_message is None:
                    error_message = f"{type(error).__name__}: {error}"
            self._buffer.close(error_message)


@dataclass(frozen=True, slots=True)
class ResolvedCameraMode:
    width: int
    height: int
    fps: float
    fourcc: str


@dataclass(frozen=True, slots=True)
class OpenCVCameraConfig:
    device_index: int | str
    width: int
    height: int
    fps: float
    backend: str = str(_LIVE_DEFAULTS["backend"])
    buffer_size: int = int(_LIVE_DEFAULTS["buffer_size"])
    fourcc: str | None = _LIVE_DEFAULTS["fourcc"]
    maximum_consecutive_read_failures: int = int(
        _LIVE_DEFAULTS["maximum_consecutive_read_failures"]
    )
    failure_retry_s: float = float(_LIVE_DEFAULTS["failure_retry_s"])

    def validate(self) -> None:
        if isinstance(self.device_index, int):
            if self.device_index < 0:
                raise ValueError("camera device index must not be negative")
        elif not isinstance(self.device_index, str) or not self.device_index.strip():
            raise ValueError("camera device must be an index or non-empty path/pipeline")
        if min(self.width, self.height) <= 0 or self.fps <= 0.0:
            raise ValueError("camera dimensions and FPS must be positive")
        if self.backend not in {"any", "avfoundation", "gstreamer", "v4l2"}:
            raise ValueError("unsupported OpenCV camera backend")
        if self.buffer_size <= 0:
            raise ValueError("camera buffer size must be positive")
        if self.fourcc is not None and len(self.fourcc) != 4:
            raise ValueError("camera FOURCC must contain four characters")
        if self.maximum_consecutive_read_failures <= 0:
            raise ValueError("maximum camera read failures must be positive")
        if self.failure_retry_s < 0.0:
            raise ValueError("camera failure retry must not be negative")


class OpenCVCameraFrameSource(_ThreadedFrameSource):
    """Headless UVC capture; this class never calls OpenCV HighGUI."""

    def __init__(self, config: OpenCVCameraConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.resolved_mode: ResolvedCameraMode | None = None
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._next_frame_id = 0
        self._consecutive_failures = 0

    @property
    def _thread_name(self) -> str:
        return "jetracer-uvc-capture"

    def _prepare(self) -> None:
        cv2 = _import_cv2()
        backend = _backend_code(cv2, self.config.backend)
        capture = (
            cv2.VideoCapture(self.config.device_index)
            if backend is None
            else cv2.VideoCapture(self.config.device_index, backend)
        )
        if not capture.isOpened():
            capture.release()
            raise FrameSourceError(
                f"cannot open camera device {self.config.device_index}"
            )
        if self.config.fourcc is not None:
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*self.config.fourcc),
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, self.config.buffer_size)
        self._cv2 = cv2
        self._capture = capture
        self.resolved_mode = ResolvedCameraMode(
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            fourcc=_decode_fourcc(round(capture.get(cv2.CAP_PROP_FOURCC))),
        )

    def _next_frame(self) -> CapturedFrame | None:
        assert self._capture is not None
        assert self._cv2 is not None
        while not self._stop_event.is_set():
            success, image_bgr = self._capture.read()
            captured_at_s = perf_counter()
            if success and image_bgr is not None:
                self._consecutive_failures = 0
                source_milliseconds = float(
                    self._capture.get(self._cv2.CAP_PROP_POS_MSEC)
                )
                source_timestamp_s = (
                    source_milliseconds / 1000.0
                    if source_milliseconds > 0.0
                    else None
                )
                frame = CapturedFrame(
                    frame_id=self._next_frame_id,
                    image_bgr=image_bgr,
                    captured_at_s=captured_at_s,
                    source_timestamp_s=source_timestamp_s,
                )
                self._next_frame_id += 1
                return frame
            self._failed_reads += 1
            self._consecutive_failures += 1
            if (
                self._consecutive_failures
                >= self.config.maximum_consecutive_read_failures
            ):
                raise FrameSourceError(
                    "camera exceeded the consecutive read-failure limit"
                )
            self._stop_event.wait(self.config.failure_retry_s)
        return None

    def _interrupt(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


@dataclass(frozen=True, slots=True)
class RecordedVideoConfig:
    path: Path
    realtime_pacing: bool = bool(_RECORDED_DEFAULTS["realtime_pacing"])
    loop: bool = bool(_RECORDED_DEFAULTS["loop"])


class RecordedVideoFrameSource(_ThreadedFrameSource):
    def __init__(self, config: RecordedVideoConfig) -> None:
        super().__init__()
        self.config = config
        self._cv2: Any | None = None
        self._capture: Any | None = None
        self._fps = 0.0
        self._media_frame_index = 0
        self._next_frame_id = 0
        self._pacing_started_at_s = 0.0

    @property
    def _thread_name(self) -> str:
        return "jetracer-recorded-capture"

    @property
    def source_fps(self) -> float:
        if self._fps <= 0.0:
            raise RuntimeError("recorded source has not been started")
        return self._fps

    def _prepare(self) -> None:
        path = Path(self.config.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"recorded video does not exist: {path}")
        cv2 = _import_cv2()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise FrameSourceError(f"cannot open recorded video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not isfinite(fps) or fps <= 0.0:
            capture.release()
            raise FrameSourceError("recorded video does not report a valid FPS")
        self._cv2 = cv2
        self._capture = capture
        self._fps = fps
        self._pacing_started_at_s = perf_counter()

    def _next_frame(self) -> CapturedFrame | None:
        assert self._capture is not None
        assert self._cv2 is not None
        success, image_bgr = self._capture.read()
        if not success or image_bgr is None:
            if not self.config.loop:
                return None
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            self._media_frame_index = 0
            self._pacing_started_at_s = perf_counter()
            success, image_bgr = self._capture.read()
            if not success or image_bgr is None:
                self._failed_reads += 1
                raise FrameSourceError("recorded video produced no frames")

        source_milliseconds = float(
            self._capture.get(self._cv2.CAP_PROP_POS_MSEC)
        )
        source_timestamp_s = (
            source_milliseconds / 1000.0
            if source_milliseconds > 0.0
            else self._media_frame_index / self._fps
        )
        if self.config.realtime_pacing:
            wait_s = (
                self._pacing_started_at_s
                + self._media_frame_index / self._fps
                - perf_counter()
            )
            if wait_s > 0.0 and self._stop_event.wait(wait_s):
                return None
        frame = CapturedFrame(
            frame_id=self._next_frame_id,
            image_bgr=image_bgr,
            captured_at_s=perf_counter(),
            source_timestamp_s=source_timestamp_s,
        )
        self._next_frame_id += 1
        self._media_frame_index += 1
        return frame

    def _interrupt(self) -> None:
        if self._capture is not None:
            self._capture.release()

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class SimulatorFrameSource(_ThreadedFrameSource):
    """Paced simulator capture using the same contract as physical cameras."""

    def __init__(
        self,
        simulator: Simulator,
        command: VehicleCommand | None = None,
    ) -> None:
        super().__init__()
        self.simulator = simulator
        self._command = command or VehicleCommand(0.0, 0.0)
        self._command_lock = Lock()
        self._first_frame: Any | None = None
        self._next_deadline_s = 0.0

    @property
    def _thread_name(self) -> str:
        return "jetracer-simulator-capture"

    def set_command(self, command: VehicleCommand) -> None:
        with self._command_lock:
            self._command = command

    def _prepare(self) -> None:
        self._first_frame = self.simulator.render_now()
        self._next_deadline_s = perf_counter()

    def _next_frame(self) -> CapturedFrame | None:
        if self._first_frame is not None:
            native_frame = self._first_frame
            self._first_frame = None
        else:
            period_s = native_camera_period(self.simulator)
            self._next_deadline_s += period_s
            wait_s = self._next_deadline_s - perf_counter()
            if wait_s > 0.0 and self._stop_event.wait(wait_s):
                return None
            with self._command_lock:
                command = self._command
            emitted = self.simulator.advance(command, period_s)
            if not emitted:
                raise FrameSourceError("simulator did not emit a camera frame")
            native_frame = emitted[-1]
        return CapturedFrame(
            frame_id=native_frame.frame_id,
            image_bgr=native_frame.to_bgr(),
            captured_at_s=perf_counter(),
            source_timestamp_s=native_frame.simulation_time_s,
            native_frame=native_frame,
        )


def native_camera_period(simulator: Simulator) -> float:
    return float(simulator.camera.frame_period_s)


def _import_cv2() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise FrameSourceError(
            "OpenCV camera/video capture requires opencv-python"
        ) from error
    return cv2


def _backend_code(cv2: Any, backend: str) -> int | None:
    names = {
        "any": None,
        "avfoundation": "CAP_AVFOUNDATION",
        "gstreamer": "CAP_GSTREAMER",
        "v4l2": "CAP_V4L2",
    }
    attribute = names[backend]
    if attribute is None:
        return None
    if not hasattr(cv2, attribute):
        raise FrameSourceError(f"OpenCV does not provide the {backend} backend")
    return int(getattr(cv2, attribute))


def _decode_fourcc(value: int) -> str:
    return "".join(chr((value >> (8 * index)) & 0xFF) for index in range(4))
