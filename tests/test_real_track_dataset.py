"""Real-track capture manifest and evaluation pipeline tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPOSITORY_ROOT / "datasets" / "real_track" / "manifest.json"


class ColourRoadAdapter(sim.SegmentationAdapter):
    @property
    def metadata(self) -> sim.ModelMetadata:
        return sim.ModelMetadata(
            model_id="colour-road-test",
            display_name="Colour road test",
            backend="numpy",
            precision="uint8",
        )

    def infer(self, image_bgr: np.ndarray) -> sim.SegmentationPrediction:
        grey = np.all(image_bgr == np.asarray((128, 128, 128)), axis=2)
        orange = np.all(image_bgr == np.asarray((0, 140, 255)), axis=2)
        return sim.SegmentationPrediction(
            labels=(grey | orange).astype(np.uint8),
            road_class_id=1,
        )


def test_empty_template_reports_awaiting_capture() -> None:
    dataset = sim.load_real_track_dataset(TEMPLATE)
    assert dataset.document["status"] == "awaiting_capture"
    assert dataset.captures == ()
    evaluation = sim.evaluate_real_track_dataset(TEMPLATE)
    assert evaluation.status == "awaiting_capture"
    assert evaluation.capture_count == 0
    assert evaluation.integrity_ready
    assert not evaluation.capture_protocol_ready
    assert not evaluation.segmentation_evaluation_ready
    calibration = sim.calibrate_real_track_colours(TEMPLATE)
    assert calibration["status"] == "awaiting_annotations"
    assert calibration["profiles"] == {}


def test_complete_dataset_calibrates_and_prepares_evaluation() -> None:
    with TemporaryDirectory(prefix="jetracer-real-track-test-") as directory:
        root = Path(directory)
        manifest = _complete_fixture(root)
        evaluation = sim.evaluate_real_track_dataset(manifest)
        assert evaluation.status == "ready"
        assert evaluation.capture_count == 3
        assert evaluation.image_count == 3
        assert evaluation.video_count == 0
        assert evaluation.annotated_image_count == 3
        assert evaluation.capture_protocol_ready
        assert evaluation.segmentation_evaluation_ready
        assert not [
            issue for issue in evaluation.issues if issue.severity == "error"
        ]

        calibration = sim.calibrate_real_track_colours(manifest)
        assert calibration["status"] == "ready"
        assert set(calibration["profiles"]) == {"1", "2", "3"}
        assert calibration["source_capture_ids"] == ["calibration-001"]
        assert calibration["evaluation"]["development"][
            "evaluated_frames"
        ] == 1
        for split in ("development", "benchmark"):
            for result in calibration["evaluation"][split]["classes"].values():
                assert result["iou"] == 1.0
                assert result["precision"] == 1.0
                assert result["recall"] == 1.0

        prepared = root / "prepared"
        prepared_manifest = sim.prepare_real_track_segmentation_dataset(
            manifest, prepared, split="benchmark"
        )
        assert prepared_manifest["frame_count"] == 1
        scored = sim.evaluate_segmentation_dataset(
            prepared,
            ColourRoadAdapter(),
            ground_truth_class_ids=(1, 2),
            warmup_frames=0,
        )
        assert scored.evaluated_frames == 1
        assert scored.road_iou == 1.0


def test_registration_fingerprints_and_tampering_is_detected() -> None:
    with TemporaryDirectory(prefix="jetracer-real-track-register-") as directory:
        root = Path(directory)
        document = _fixture_document()
        document["captures"] = []
        document["status"] = "awaiting_capture"
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(document), encoding="utf-8")
        media = root / "media"
        annotations = root / "annotations"
        media.mkdir()
        annotations.mkdir()
        image_path, mask_path = _write_labelled_image(media, annotations, "one")
        capture = sim.register_real_track_capture(
            manifest,
            capture_id="registered-001",
            media_path=image_path,
            split="calibration",
            media_type="image",
            camera_mode_id="test_6x4",
            lighting_condition="test_light",
            track_section="straight",
            scene_type="empty_road",
            semantic_mask_path=mask_path,
        )
        assert len(capture["sha256"]) == 64
        assert len(capture["semantic_mask_sha256"]) == 64
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        assert saved["status"] == "collecting"
        assert saved["captures"][0]["capture_id"] == "registered-001"

        image_path.write_bytes(b"tampered")
        evaluation = sim.evaluate_real_track_dataset(
            manifest, probe_media=False
        )
        codes = {issue.code for issue in evaluation.issues}
        assert "media_sha256_mismatch" in codes
        assert evaluation.status == "invalid"


def test_unsafe_paths_and_gui_calls_are_rejected() -> None:
    with TemporaryDirectory(prefix="jetracer-real-track-path-") as directory:
        root = Path(directory)
        document = _fixture_document()
        document["captures"] = [
            {
                "capture_id": "unsafe-001",
                "split": "calibration",
                "media_type": "image",
                "path": "../outside.png",
                "sha256": "0" * 64,
                "camera_mode_id": "test_6x4",
                "lighting_condition": "test_light",
                "track_section": "straight",
                "scene_type": "empty_road"
            }
        ]
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps(document), encoding="utf-8")
        evaluation = sim.evaluate_real_track_dataset(
            manifest, probe_media=False
        )
        assert "unsafe_path" in {issue.code for issue in evaluation.issues}

    for tool_name in (
        "evaluate_real_track_dataset.py",
        "register_real_track_capture.py",
        "prepare_real_track_segmentation.py",
        "calibrate_real_track_colours.py",
    ):
        source = (REPOSITORY_ROOT / "tools" / tool_name).read_text(
            encoding="utf-8"
        )
        for forbidden in ("namedWindow", "imshow", "waitKey", "webbrowser"):
            assert forbidden not in source


def _complete_fixture(root: Path) -> Path:
    document = _fixture_document()
    media = root / "media"
    annotations = root / "annotations"
    media.mkdir()
    annotations.mkdir()
    captures = []
    for split in sim.REAL_TRACK_SPLITS:
        image_path, mask_path = _write_labelled_image(
            media, annotations, split
        )
        captures.append(
            {
                "capture_id": f"{split}-001",
                "split": split,
                "media_type": "image",
                "path": image_path.relative_to(root).as_posix(),
                "sha256": sim.real_track_file_sha256(image_path),
                "camera_mode_id": "test_6x4",
                "lighting_condition": "test_light",
                "track_section": "straight",
                "scene_type": "empty_road",
                "semantic_mask_path": mask_path.relative_to(root).as_posix(),
                "semantic_mask_sha256": sim.real_track_file_sha256(mask_path),
            }
        )
    document["captures"] = captures
    document["status"] = "frozen"
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    return manifest


def _fixture_document() -> dict[str, object]:
    document = deepcopy(json.loads(TEMPLATE.read_text(encoding="utf-8")))
    document["camera_modes"] = [
        {
            "camera_mode_id": "test_6x4",
            "camera_id": "test",
            "width": 6,
            "height": 4,
            "fps": 1.0,
            "pixel_format": "rgb",
        }
    ]
    protocol = document["capture_protocol"]
    protocol["lighting_conditions"] = ["test_light"]
    protocol["track_sections"] = ["straight"]
    protocol["scene_types"] = ["empty_road"]
    protocol["required_lighting_conditions"] = ["test_light"]
    protocol["required_track_sections"] = ["straight"]
    protocol["required_scene_types"] = ["empty_road"]
    protocol["minimum_captures_per_split"] = {
        split: 1 for split in sim.REAL_TRACK_SPLITS
    }
    protocol["minimum_annotated_benchmark_images"] = 1
    return document


def _write_labelled_image(
    media: Path, annotations: Path, stem: str
) -> tuple[Path, Path]:
    semantic = np.asarray(
        (
            (0, 0, 0, 0, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (2, 2, 2, 2, 2, 2),
            (3, 3, 3, 3, 3, 3),
        ),
        dtype=np.uint8,
    )
    colours = np.asarray(
        (
            (0, 0, 0),
            (128, 128, 128),
            (255, 140, 0),
            (255, 0, 255),
        ),
        dtype=np.uint8,
    )
    rgb = colours[semantic]
    image_path = media / f"{stem}.png"
    mask_path = annotations / f"{stem}-semantic.png"
    Image.fromarray(rgb).save(image_path)
    Image.fromarray(semantic).save(mask_path)
    return image_path, mask_path


def main() -> None:
    test_empty_template_reports_awaiting_capture()
    test_complete_dataset_calibrates_and_prepares_evaluation()
    test_registration_fingerprints_and_tampering_is_detected()
    test_unsafe_paths_and_gui_calls_are_rejected()


if __name__ == "__main__":
    main()
