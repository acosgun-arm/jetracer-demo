"""Tests for memory-bounded video lane calibration workflow."""

from __future__ import annotations

from json import dumps, loads
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

import jetracer_sim as sim


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "video_lane_calibration.json"
TEMPLATE = ROOT / "configs" / "color_lane" / "waveshare-sim-white.json"


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], fps: float) -> None:
        self.frames = frames
        self.fps = fps
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, field: int) -> float:
        values = {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.frames),
            cv2.CAP_PROP_FRAME_WIDTH: self.frames[0].shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self.frames[0].shape[0],
        }
        return float(values[field])

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, np.array(frame, copy=True)

    def release(self) -> None:
        pass


class FakeCv2:
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
    CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    IMWRITE_PNG_COMPRESSION = cv2.IMWRITE_PNG_COMPRESSION
    INTER_AREA = cv2.INTER_AREA
    COLOR_BGR2HSV = cv2.COLOR_BGR2HSV
    COLOR_BGR2GRAY = cv2.COLOR_BGR2GRAY
    COLOR_BGR2LAB = cv2.COLOR_BGR2LAB
    CV_64F = cv2.CV_64F
    TERM_CRITERIA_EPS = cv2.TERM_CRITERIA_EPS
    TERM_CRITERIA_COUNT = cv2.TERM_CRITERIA_COUNT
    INTER_NEAREST = cv2.INTER_NEAREST

    def __init__(self, frames: list[np.ndarray], fps: float) -> None:
        self.frames = frames
        self.fps = fps

    def VideoCapture(self, _path: str) -> FakeCapture:  # noqa: N802
        return FakeCapture(self.frames, self.fps)

    resize = staticmethod(cv2.resize)
    cvtColor = staticmethod(cv2.cvtColor)
    calcHist = staticmethod(cv2.calcHist)
    Laplacian = staticmethod(cv2.Laplacian)
    imwrite = staticmethod(cv2.imwrite)
    imread = staticmethod(cv2.imread)
    calcOpticalFlowPyrLK = staticmethod(cv2.calcOpticalFlowPyrLK)
    fillPoly = staticmethod(cv2.fillPoly)


def test_diverse_workspace_and_sparse_calibration() -> None:
    config = sim.load_video_lane_calibration_config(CONFIG)
    config["keyframes"]["selected_frame_count"] = 5
    config["keyframes"]["sample_interval_s"] = 0.1
    config["keyframes"]["minimum_separation_s"] = 0.1
    config["annotations"]["sample_radius_pixels"] = 1
    frames = []
    for index in range(12):
        image = np.full((30, 40, 3), 30 + index * 8, np.uint8)
        image[:, 8:12] = (0, 165, 255)
        image[:, 28:32] = (0, 165, 255)
        frames.append(image)
    fake_cv2 = FakeCv2(frames, 10.0)
    with TemporaryDirectory(prefix="jetracer-video-calibration-") as directory:
        root = Path(directory)
        video = root / "source.mp4"
        video.write_bytes(b"deterministic fake video")
        workspace_root = root / "workspace"
        workspace = sim.prepare_video_lane_workspace(
            video,
            workspace_root,
            config,
            cv2=fake_cv2,
            track_profile_id="waveshare",
            camera_profile_id="elp_112",
        )
        assert workspace["selection"]["selected_frame_count"] == 5
        assert len(list((workspace_root / "frames").glob("*.png"))) == 5
        annotations = sim.load_video_lane_annotations(
            workspace_root / "annotations.json"
        )
        for frame in workspace["frames"][:2]:
            annotations["frames"][frame["frame_id"]] = {
                "status": "annotated",
                "lane_points": [[0.24, 0.5], [0.75, 0.5]],
                "background_points": [[0.5, 0.5], [0.5, 0.2]],
                "left_polyline": [[0.24, 0.1], [0.24, 0.9]],
                "right_polyline": [[0.75, 0.1], [0.75, 0.9]],
                "road_polygon": [
                    [0.28, 0.0],
                    [0.72, 0.0],
                    [0.72, 1.0],
                    [0.28, 1.0],
                ],
            }
        sim.save_video_lane_annotations(
            workspace_root / "annotations.json", annotations
        )
        calibration = sim.calibrate_sparse_lane_colours(
            workspace_root / "workspace.json",
            workspace_root / "annotations.json",
            config,
            cv2=cv2,
        )
        assert calibration["status"] == "calibrated"
        assert calibration["lane_recall"] >= 0.98
        assert calibration["background_false_positive_rate"] == 0.0
        output = root / "profile.json"
        sim.export_calibrated_color_lane_profile(
            TEMPLATE,
            calibration,
            output,
            profile_id="waveshare-elp-test",
        )
        profile = loads(output.read_text(encoding="utf-8"))
        assert profile["profile_id"] == "waveshare-elp-test"
        assert profile["calibration_provenance"]["method"] == (
            "sparse_video_keyframes"
        )
        sim.load_color_lane_profile(output)
        benchmark = sim.benchmark_video_lane_pixel_masks(
            workspace_root / "workspace.json",
            workspace_root / "annotations.json",
            output,
            config,
            cv2=cv2,
        )
        assert benchmark["status"] == "complete"
        assert benchmark["frame_count"] == 2
        assert 0.0 <= benchmark["iou"] <= 1.0

        review = sim.rank_video_lane_review_frames(
            workspace,
            sim.load_video_lane_annotations(workspace_root / "annotations.json"),
            config,
        )
        assert len(review) == len(workspace["frames"])
        assert review == sorted(review, key=lambda item: (-item["score"], item["frame_id"]))


def test_optical_flow_creates_reviewable_proposals_only() -> None:
    config = sim.load_video_lane_calibration_config(CONFIG)
    config["keyframes"]["selected_frame_count"] = 6
    config["keyframes"]["sample_interval_s"] = 0.1
    config["keyframes"]["minimum_separation_s"] = 0.0
    config["optical_flow"]["minimum_retained_fraction"] = 0.4
    frames: list[np.ndarray] = []
    for index in range(8):
        image = np.zeros((80, 120, 3), np.uint8)
        cv2.rectangle(image, (15 + index, 10), (25 + index, 70), (0, 165, 255), -1)
        cv2.rectangle(image, (85 + index, 10), (95 + index, 70), (0, 165, 255), -1)
        for y in range(15, 70, 10):
            cv2.circle(image, (20 + index, y), 2, (255, 255, 255), -1)
            cv2.circle(image, (90 + index, y), 2, (255, 255, 255), -1)
        frames.append(image)
    fake_cv2 = FakeCv2(frames, 10.0)
    with TemporaryDirectory(prefix="jetracer-flow-propagation-") as directory:
        root = Path(directory)
        video = root / "source.mp4"
        video.write_bytes(b"fake moving video")
        workspace_root = root / "workspace"
        workspace = sim.prepare_video_lane_workspace(
            video,
            workspace_root,
            config,
            cv2=fake_cv2,
            track_profile_id="waveshare",
            camera_profile_id="elp_112",
        )
        annotations = sim.load_video_lane_annotations(workspace_root / "annotations.json")
        first = workspace["frames"][0]["frame_id"]
        annotations["frames"][first] = {
            "status": "annotated",
            "lane_points": [[0.17, 0.25], [0.75, 0.25]],
            "background_points": [[0.5, 0.5]],
            "left_polyline": [[0.17, 0.2], [0.17, 0.8]],
            "right_polyline": [[0.75, 0.2], [0.75, 0.8]],
            "road_polygon": [[0.22, 0.1], [0.7, 0.1], [0.7, 0.9], [0.22, 0.9]],
        }
        sim.save_video_lane_annotations(workspace_root / "annotations.json", annotations)
        report = sim.propagate_video_lane_annotations(
            workspace_root / "workspace.json",
            workspace_root / "annotations.json",
            config,
            cv2=fake_cv2,
        )
        saved = sim.load_video_lane_annotations(workspace_root / "annotations.json")
        assert report["proposals_created"] > 0
        assert saved["frames"][first]["status"] == "annotated"
        proposals = [value for value in saved["frames"].values() if value["status"] == "proposed"]
        assert proposals
        assert all(0.0 <= item["propagation"]["confidence"] <= 1.0 for item in proposals)


def test_annotation_validation_rejects_non_normalized_coordinates() -> None:
    invalid = {
        "schema_version": sim.VIDEO_LANE_ANNOTATION_SCHEMA_VERSION,
        "frames": {
            "frame-1": {
                "status": "annotated",
                "lane_points": [[1.2, 0.5]],
            }
        },
    }
    try:
        sim.validate_video_lane_annotations(invalid)
    except ValueError as error:
        assert "normalized" in str(error)
    else:
        raise AssertionError("invalid coordinate was accepted")


def main() -> None:
    test_diverse_workspace_and_sparse_calibration()
    test_optical_flow_creates_reviewable_proposals_only()
    test_annotation_validation_rejects_non_normalized_coordinates()


if __name__ == "__main__":
    main()
