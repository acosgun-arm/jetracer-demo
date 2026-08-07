"""Asynchronous raw-camera capture for the browser demo."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread, current_thread
from typing import Any, Mapping

import numpy as np

from .frame_source import CapturedFrame
from .real_track_dataset import (
    RealTrackDataset,
    load_real_track_dataset,
    register_real_track_capture,
)


CAPTURE_ACTIONS = ("snapshot", "start_video", "stop_video")
CAPTURE_METADATA_FIELDS = (
    "split",
    "lighting_condition",
    "track_section",
    "scene_type",
)


class RealTrackCaptureManager:
    """Save raw frames without blocking the camera/control loop."""

    def __init__(
        self,
        cv2: Any,
        manifest_path: str | Path,
        camera_mode_id: str,
        options: Mapping[str, Any],
    ) -> None:
        self._cv2 = cv2
        self._dataset = load_real_track_dataset(manifest_path)
        self._camera_mode_id = camera_mode_id
        self._mode = _camera_mode(self._dataset, camera_mode_id)
        self._options = _validated_options(options)
        self._media_directory = self._dataset.root / "media"
        self._media_directory.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._manifest_lock = Lock()
        self._snapshot_threads: set[Thread] = set()
        self._video_thread: Thread | None = None
        self._video_queue: Queue[np.ndarray] | None = None
        self._video_stop = Event()
        self._video_writer: Any | None = None
        self._video_capture_id: str | None = None
        self._video_path: Path | None = None
        self._video_metadata: dict[str, str] | None = None
        self._status: dict[str, Any] = {
            "state": "idle",
            "message": "Ready to capture raw camera frames",
            "active_capture_id": None,
            "queued_frames": 0,
            "written_frames": 0,
            "dropped_frames": 0,
            "pending_snapshots": 0,
            "last_capture": None,
        }

    @property
    def catalog(self) -> dict[str, Any]:
        protocol = self._dataset.protocol
        return {
            "enabled": True,
            "track_id": str(self._dataset.document["track"]["track_id"]),
            "track_name": str(
                self._dataset.document["track"].get(
                    "display_name", self._dataset.document["track"]["track_id"]
                )
            ),
            "camera_mode_id": self._camera_mode_id,
            "camera_mode": deepcopy(self._mode),
            "splits": list(protocol["minimum_captures_per_split"]),
            "lighting_conditions": list(protocol["lighting_conditions"]),
            "track_sections": list(protocol["track_sections"]),
            "scene_types": list(protocol["scene_types"]),
            "raw_frames": True,
        }

    @property
    def status(self) -> dict[str, Any]:
        with self._lock:
            result = deepcopy(self._status)
            queue = self._video_queue
            result["queued_frames"] = 0 if queue is None else queue.qsize()
            return result

    def handle_request(
        self,
        request: Mapping[str, Any],
        frame: CapturedFrame,
    ) -> None:
        try:
            action = str(request.get("action", ""))
            if action not in CAPTURE_ACTIONS:
                raise ValueError(f"unsupported capture action: {action!r}")
            if action == "stop_video":
                self.stop_video()
                return
            metadata = _capture_metadata(self._dataset, request)
            _validate_frame(frame.image_bgr, self._mode)
            if action == "snapshot":
                self.snapshot(frame.image_bgr, metadata)
            else:
                self.start_video(frame.image_bgr, metadata)
        except (OSError, RuntimeError, ValueError) as error:
            self._set_error(str(error))

    def snapshot(
        self,
        image_bgr: np.ndarray,
        metadata: Mapping[str, str],
    ) -> None:
        with self._lock:
            if len(self._snapshot_threads) >= int(
                self._options["maximum_pending_snapshots"]
            ):
                raise RuntimeError("too many snapshots are already being saved")
            capture_id = self._new_capture_id("image", metadata)
            path = self._media_directory / (
                capture_id + str(self._options["image_extension"])
            )
            image = np.array(image_bgr, copy=True)
            thread = Thread(
                target=self._save_snapshot,
                args=(capture_id, path, image, dict(metadata)),
                name=f"jetracer-snapshot-{capture_id}",
                daemon=True,
            )
            self._snapshot_threads.add(thread)
            self._status["pending_snapshots"] = len(self._snapshot_threads)
            self._status["message"] = f"Saving snapshot {capture_id}"
        thread.start()

    def start_video(
        self,
        first_image_bgr: np.ndarray,
        metadata: Mapping[str, str],
    ) -> None:
        with self._lock:
            if self._status["state"] in {"recording", "finalizing"}:
                raise RuntimeError("a video capture is already active")
            capture_id = self._new_capture_id("video", metadata)
            path = self._media_directory / (
                capture_id + str(self._options["video_extension"])
            )
            fourcc = self._cv2.VideoWriter_fourcc(
                *str(self._options["video_fourcc"])
            )
            writer = self._cv2.VideoWriter(
                str(path),
                fourcc,
                float(self._mode["fps"]),
                (int(self._mode["width"]), int(self._mode["height"])),
            )
            if not writer.isOpened():
                writer.release()
                raise RuntimeError(
                    "video writer could not open; check the configured codec"
                )
            self._video_writer = writer
            self._video_queue = Queue(
                maxsize=int(self._options["writer_queue_frames"])
            )
            self._video_stop.clear()
            self._video_capture_id = capture_id
            self._video_path = path
            self._video_metadata = dict(metadata)
            self._status.update(
                {
                    "state": "recording",
                    "message": f"Recording {capture_id}",
                    "active_capture_id": capture_id,
                    "queued_frames": 0,
                    "written_frames": 0,
                    "dropped_frames": 0,
                }
            )
            self._video_thread = Thread(
                target=self._write_video,
                name=f"jetracer-video-{capture_id}",
                daemon=True,
            )
            thread = self._video_thread
        thread.start()
        self.record_frame(first_image_bgr)

    def record_frame(self, frame: CapturedFrame | np.ndarray) -> None:
        image_bgr = frame.image_bgr if isinstance(frame, CapturedFrame) else frame
        with self._lock:
            if self._status["state"] != "recording":
                return
            queue = self._video_queue
        assert queue is not None
        try:
            queue.put_nowait(image_bgr)
        except Full:
            with self._lock:
                self._status["dropped_frames"] += 1

    def stop_video(self) -> None:
        with self._lock:
            if self._status["state"] == "recording":
                self._status["state"] = "finalizing"
                self._status["message"] = "Finalizing video and manifest entry"
                self._video_stop.set()
            elif self._status["state"] == "finalizing":
                return
            else:
                raise RuntimeError("no video capture is active")

    def close(self) -> None:
        with self._lock:
            recording = self._status["state"] == "recording"
            video_thread = self._video_thread
            snapshot_threads = tuple(self._snapshot_threads)
        if recording:
            self.stop_video()
        if video_thread is not None:
            video_thread.join(float(self._options["writer_stop_timeout_s"]))
            if video_thread.is_alive():
                self._set_error(
                    "video writer did not stop before its timeout",
                    preserve_active=False,
                )
        for thread in snapshot_threads:
            thread.join(float(self._options["snapshot_stop_timeout_s"]))
            if thread.is_alive():
                self._set_error(
                    "snapshot writer did not stop before its timeout",
                    preserve_active=False,
                )

    def _save_snapshot(
        self,
        capture_id: str,
        path: Path,
        image_bgr: np.ndarray,
        metadata: Mapping[str, str],
    ) -> None:
        current = current_thread()
        try:
            parameters = [
                int(self._cv2.IMWRITE_PNG_COMPRESSION),
                int(self._options["png_compression"]),
            ]
            if not self._cv2.imwrite(str(path), image_bgr, parameters):
                raise RuntimeError("OpenCV failed to write the snapshot")
            capture = self._register(
                capture_id, path, "image", metadata
            )
            with self._lock:
                self._status["last_capture"] = capture
                self._status["message"] = f"Saved snapshot {capture_id}"
                if self._status["state"] == "error":
                    self._status["state"] = "idle"
        except (OSError, RuntimeError, ValueError) as error:
            path.unlink(missing_ok=True)
            self._set_error(str(error))
        finally:
            with self._lock:
                self._snapshot_threads.discard(current)
                self._status["pending_snapshots"] = len(
                    self._snapshot_threads
                )

    def _write_video(self) -> None:
        with self._lock:
            queue = self._video_queue
            writer = self._video_writer
            capture_id = self._video_capture_id
            path = self._video_path
            metadata = self._video_metadata
        assert queue is not None
        assert writer is not None
        assert capture_id is not None
        assert path is not None
        assert metadata is not None
        try:
            while not self._video_stop.is_set() or not queue.empty():
                try:
                    image_bgr = queue.get(
                        timeout=float(self._options["writer_poll_timeout_s"])
                    )
                except Empty:
                    continue
                writer.write(image_bgr)
                with self._lock:
                    self._status["written_frames"] += 1
            writer.release()
            with self._lock:
                written_frames = int(self._status["written_frames"])
                dropped_frames = int(self._status["dropped_frames"])
            if written_frames == 0:
                raise RuntimeError("video capture contained no frames")
            capture = self._register(
                capture_id,
                path,
                "video",
                metadata,
                capture_details={
                    "target_fps": float(self._mode["fps"]),
                    "written_frames": written_frames,
                    "dropped_frames": dropped_frames,
                    "video_fourcc": str(self._options["video_fourcc"]),
                },
            )
            with self._lock:
                self._status.update(
                    {
                        "state": "idle",
                        "message": (
                            f"Saved video {capture_id}; "
                            f"{dropped_frames} frames dropped"
                        ),
                        "active_capture_id": None,
                        "last_capture": capture,
                    }
                )
        except (OSError, RuntimeError, ValueError) as error:
            writer.release()
            path.unlink(missing_ok=True)
            self._set_error(str(error), preserve_active=False)
        finally:
            with self._lock:
                self._video_writer = None
                self._video_queue = None
                self._video_capture_id = None
                self._video_path = None
                self._video_metadata = None
                self._video_thread = None

    def _register(
        self,
        capture_id: str,
        path: Path,
        media_type: str,
        metadata: Mapping[str, str],
        capture_details: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._manifest_lock:
            return register_real_track_capture(
                self._dataset.manifest_path,
                capture_id=capture_id,
                media_path=path,
                split=metadata["split"],
                media_type=media_type,
                camera_mode_id=self._camera_mode_id,
                lighting_condition=metadata["lighting_condition"],
                track_section=metadata["track_section"],
                scene_type=metadata["scene_type"],
                capture_details=capture_details,
            )

    def _new_capture_id(
        self,
        media_type: str,
        metadata: Mapping[str, str],
    ) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return "-".join(
            (
                timestamp,
                _slug(metadata["split"]),
                _slug(metadata["track_section"]),
                _slug(metadata["scene_type"]),
                media_type,
            )
        )

    def _set_error(
        self,
        message: str,
        *,
        preserve_active: bool = True,
    ) -> None:
        with self._lock:
            if not preserve_active or self._status["state"] not in {
                "recording",
                "finalizing",
            }:
                self._status["state"] = "error"
            self._status["message"] = message


def _camera_mode(dataset: RealTrackDataset, camera_mode_id: str) -> dict[str, Any]:
    mode = next(
        (
            item
            for item in dataset.document["camera_modes"]
            if item.get("camera_mode_id") == camera_mode_id
        ),
        None,
    )
    if mode is None:
        raise ValueError(f"unknown real-track camera mode: {camera_mode_id}")
    return deepcopy(mode)


def _capture_metadata(
    dataset: RealTrackDataset,
    request: Mapping[str, Any],
) -> dict[str, str]:
    protocol = dataset.protocol
    allowed = {
        "split": set(protocol["minimum_captures_per_split"]),
        "lighting_condition": set(protocol["lighting_conditions"]),
        "track_section": set(protocol["track_sections"]),
        "scene_type": set(protocol["scene_types"]),
    }
    metadata: dict[str, str] = {}
    for field in CAPTURE_METADATA_FIELDS:
        value = request.get(field)
        if not isinstance(value, str) or value not in allowed[field]:
            raise ValueError(f"invalid capture {field.replace('_', ' ')}")
        metadata[field] = value
    return metadata


def _validate_frame(image_bgr: np.ndarray, mode: Mapping[str, Any]) -> None:
    expected = (int(mode["height"]), int(mode["width"]), 3)
    if image_bgr.shape != expected or image_bgr.dtype != np.uint8:
        raise ValueError(
            f"camera frame is {image_bgr.shape}/{image_bgr.dtype}; "
            f"capture mode requires {expected}/uint8"
        )


def _validated_options(options: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "image_extension",
        "png_compression",
        "video_extension",
        "video_fourcc",
        "writer_queue_frames",
        "writer_poll_timeout_s",
        "writer_stop_timeout_s",
        "snapshot_stop_timeout_s",
        "maximum_pending_snapshots",
        "maximum_request_bytes",
    }
    missing = required - set(options)
    if missing:
        raise ValueError(
            "real-track capture configuration is incomplete: "
            + ", ".join(sorted(missing))
        )
    result = dict(options)
    if result["image_extension"] != ".png":
        raise ValueError("browser snapshots currently require PNG output")
    if str(result["video_extension"]) not in {".mp4", ".mov", ".mkv"}:
        raise ValueError("unsupported capture video extension")
    if len(str(result["video_fourcc"])) != 4:
        raise ValueError("capture video FOURCC must contain four characters")
    if not 0 <= int(result["png_compression"]) <= 9:
        raise ValueError("PNG compression must be in [0, 9]")
    positive_integer_fields = (
        "writer_queue_frames",
        "maximum_pending_snapshots",
        "maximum_request_bytes",
    )
    if any(int(result[field]) <= 0 for field in positive_integer_fields):
        raise ValueError("capture queue/request limits must be positive")
    positive_time_fields = (
        "writer_poll_timeout_s",
        "writer_stop_timeout_s",
        "snapshot_stop_timeout_s",
    )
    if any(float(result[field]) <= 0.0 for field in positive_time_fields):
        raise ValueError("capture timeouts must be positive")
    return result


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in value.lower()
    ).strip("-")
