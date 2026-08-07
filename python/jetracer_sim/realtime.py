"""Asynchronous latest-frame inference for real-time control loops."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from threading import Condition, Thread
from time import perf_counter, sleep
from typing import Generic, TypeVar

import numpy as np

from .configuration import runtime_config_section
from .detection import DetectionPipeline, TimedDetections
from .frame_source import CapturedFrame
from .inference import (
    ModelMetadata,
    SegmentationAdapter,
    SegmentationPipeline,
    SegmentationPrediction,
    TimedSegmentation,
)


_DEFAULTS = runtime_config_section("realtime_worker")


class SemanticMaskSegmentationAdapter(SegmentationAdapter):
    """Treat channel zero as simulator semantic labels.

    This adapter is for scheduling and control experiments using perfect
    simulator perception. It deliberately is not a deployable vision model.
    """

    def __init__(self) -> None:
        self._metadata = ModelMetadata(
            model_id="simulator-semantic",
            display_name="Simulator semantic labels",
            backend="simulator",
            precision="uint8",
            compression="ground-truth",
        )

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        labels = np.array(image_bgr[:, :, 0], dtype=np.uint8, copy=True)
        return SegmentationPrediction(
            labels=labels,
            road_class_id=int(_DEFAULTS["simulator_semantic_road_class_id"]),
        )


class LatencyInjectedSegmentationAdapter(SegmentationAdapter):
    """Wrap an adapter and enforce a minimum wall-clock inference latency."""

    def __init__(
        self,
        adapter: SegmentationAdapter,
        metadata: ModelMetadata,
        minimum_latency_s: float,
    ) -> None:
        if minimum_latency_s < 0.0:
            raise ValueError("minimum latency must not be negative")
        self._adapter = adapter
        self._metadata = metadata
        self.minimum_latency_s = minimum_latency_s

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    def infer(self, image_bgr: np.ndarray) -> SegmentationPrediction:
        started_at = perf_counter()
        prediction = self._adapter.infer(image_bgr)
        remaining_s = self.minimum_latency_s - (perf_counter() - started_at)
        if remaining_s > 0.0:
            sleep(remaining_s)
        return prediction


@dataclass(frozen=True, slots=True)
class InferenceWorkerStatistics:
    submitted_frames: int
    completed_frames: int
    replaced_pending_frames: int
    discarded_results: int
    failed_frames: int
    pending: bool
    last_error: str | None
    rate_limited_frames: int = 0


@dataclass(frozen=True, slots=True)
class _WorkItem:
    image_bgr: np.ndarray
    frame_id: int
    captured_at_s: float
    epoch: int


_TimedResult = TypeVar("_TimedResult", TimedSegmentation, TimedDetections)


class _LatestFrameInferenceWorker(Generic[_TimedResult]):
    """Run one inference pipeline with a one-frame input mailbox."""

    def __init__(
        self,
        pipeline: SegmentationPipeline | DetectionPipeline,
        *,
        worker_name: str,
        maximum_submission_fps: float | None = None,
    ) -> None:
        if maximum_submission_fps is not None and (
            not isfinite(maximum_submission_fps)
            or maximum_submission_fps <= 0.0
        ):
            raise ValueError("maximum submission FPS must be positive")
        self.pipeline = pipeline
        self._worker_name = worker_name
        self._submission_interval_s = (
            None
            if maximum_submission_fps is None
            else 1.0 / maximum_submission_fps
        )
        self._condition = Condition()
        self._thread: Thread | None = None
        self._running = False
        self._pending: _WorkItem | None = None
        self._latest: _TimedResult | None = None
        self._epoch = 0
        self._submitted_frames = 0
        self._completed_frames = 0
        self._replaced_pending_frames = 0
        self._discarded_results = 0
        self._failed_frames = 0
        self._rate_limited_frames = 0
        self._last_error: str | None = None
        self._last_submission_timestamp_s: float | None = None
        self._next_submission_timestamp_s: float | None = None

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = Thread(
                target=self._run,
                name=f"jetracer-{self._worker_name}",
                daemon=True,
            )
            self._thread.start()

    def stop(
        self,
        timeout_s: float | None = float(_DEFAULTS["stop_timeout_s"]),
    ) -> None:
        with self._condition:
            if not self._running:
                return
            self._running = False
            self._pending = None
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                raise TimeoutError(f"{self._worker_name} worker did not stop")
        with self._condition:
            self._thread = None

    def __enter__(self) -> _LatestFrameInferenceWorker[_TimedResult]:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def submit(
        self,
        image_bgr: np.ndarray,
        *,
        frame_id: int,
        captured_at_s: float | None = None,
    ) -> bool:
        captured_at = perf_counter() if captured_at_s is None else captured_at_s
        with self._condition:
            if not self._running:
                raise RuntimeError(f"{self._worker_name} worker is not running")
            if not self._submission_is_due(captured_at):
                self._rate_limited_frames += 1
                return False
            self._submitted_frames += 1
            if self._pending is not None:
                self._replaced_pending_frames += 1
            self._pending = _WorkItem(
                image_bgr=image_bgr,
                frame_id=frame_id,
                captured_at_s=captured_at,
                epoch=self._epoch,
            )
            self._condition.notify()
            return True

    def submit_captured_frame(self, frame: CapturedFrame) -> None:
        """Submit a frame without losing its capture timestamp."""

        self.submit(
            frame.image_bgr,
            frame_id=frame.frame_id,
            captured_at_s=frame.captured_at_s,
        )

    def switch_model(self, model_id: str) -> None:
        """Atomically switch and invalidate pending or in-flight old work."""

        if model_id == self.pipeline.active_model_id:
            return
        self.pipeline.switch_model(model_id)
        self.clear_results()

    def clear_results(self) -> None:
        """Invalidate work after a simulator reset or other discontinuity."""

        with self._condition:
            self._epoch += 1
            if self._pending is not None:
                self._discarded_results += 1
            self._pending = None
            self._latest = None
            self._last_submission_timestamp_s = None
            self._next_submission_timestamp_s = None
            self._condition.notify_all()

    @property
    def latest_result(self) -> _TimedResult | None:
        with self._condition:
            return self._latest

    @property
    def statistics(self) -> InferenceWorkerStatistics:
        with self._condition:
            return InferenceWorkerStatistics(
                submitted_frames=self._submitted_frames,
                completed_frames=self._completed_frames,
                replaced_pending_frames=self._replaced_pending_frames,
                discarded_results=self._discarded_results,
                failed_frames=self._failed_frames,
                pending=self._pending is not None,
                last_error=self._last_error,
                rate_limited_frames=self._rate_limited_frames,
            )

    def _submission_is_due(self, captured_at_s: float) -> bool:
        interval_s = self._submission_interval_s
        if interval_s is None:
            return True
        if (
            self._last_submission_timestamp_s is None
            or captured_at_s < self._last_submission_timestamp_s
            or self._next_submission_timestamp_s is None
        ):
            self._last_submission_timestamp_s = captured_at_s
            self._next_submission_timestamp_s = captured_at_s + interval_s
            return True
        self._last_submission_timestamp_s = captured_at_s
        if captured_at_s + 1e-12 < self._next_submission_timestamp_s:
            return False
        elapsed_intervals = floor(
            (captured_at_s - self._next_submission_timestamp_s) / interval_s
        )
        self._next_submission_timestamp_s += (
            elapsed_intervals + 1
        ) * interval_s
        return True

    def wait_for_result(
        self,
        *,
        minimum_frame_id: int | None = None,
        timeout_s: float = float(_DEFAULTS["result_timeout_s"]),
    ) -> _TimedResult | None:
        deadline = perf_counter() + timeout_s
        with self._condition:
            while self._running:
                if self._latest is not None and (
                    minimum_frame_id is None
                    or self._latest.metrics.frame_id >= minimum_frame_id
                ):
                    return self._latest
                remaining_s = deadline - perf_counter()
                if remaining_s <= 0.0:
                    return None
                self._condition.wait(remaining_s)
            return self._latest

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._running and self._pending is None:
                    self._condition.wait()
                if not self._running:
                    return
                item = self._pending
                self._pending = None
            if item is None:
                continue

            try:
                result = self.pipeline.infer(
                    item.image_bgr,
                    frame_id=item.frame_id,
                    captured_at_s=item.captured_at_s,
                )
            except Exception as error:  # Keep the safety loop alive.
                with self._condition:
                    self._failed_frames += 1
                    self._last_error = f"{type(error).__name__}: {error}"
                    self._condition.notify_all()
                continue

            with self._condition:
                current_generation = self.pipeline.model_generation
                accepted = (
                    item.epoch == self._epoch
                    and result.metrics.model_generation == current_generation
                )
                if accepted:
                    self._latest = result
                    self._completed_frames += 1
                    self._last_error = None
                else:
                    self._discarded_results += 1
                self._condition.notify_all()


class LatestFrameSegmentationWorker(
    _LatestFrameInferenceWorker[TimedSegmentation]
):
    """Run segmentation independently with latest-frame replacement."""

    def __init__(self, pipeline: SegmentationPipeline) -> None:
        super().__init__(pipeline, worker_name="segmentation")


class LatestFrameDetectionWorker(_LatestFrameInferenceWorker[TimedDetections]):
    """Run object detection independently with latest-frame replacement."""

    def __init__(
        self,
        pipeline: DetectionPipeline,
        *,
        maximum_submission_fps: float | None = None,
    ) -> None:
        super().__init__(
            pipeline,
            worker_name="detection",
            maximum_submission_fps=maximum_submission_fps,
        )
