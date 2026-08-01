"""Integrity checks for the portable evaluation dataset exporter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import jetracer_sim as sim


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="jetracer-dataset-test-"
    ) as temporary_directory:
        output_dir = Path(temporary_directory) / "evaluation-data"
        camera = sim.CameraProfile.stress_720p_200()
        camera.width = 160
        camera.height = 90
        camera.apply_nominal_intrinsics()
        config = sim.DatasetExportConfig(
            output_dir=output_dir,
            camera=camera,
            scene_count=1,
            frames_per_scene=3,
            first_seed=42,
            sample_fps=20.0,
            cruise_speed_mps=0.8,
            obstacle_count=2,
            stop_sign_count=2,
            image_format="png",
        )
        progress: list[tuple[int, int]] = []
        summary = sim.export_evaluation_dataset(
            config,
            progress=lambda completed, total: progress.append((completed, total)),
        )

        assert summary.output_dir == output_dir.resolve()
        assert summary.scene_count == 1
        assert summary.frame_count == 3
        assert progress == [(1, 3), (2, 3), (3, 3)]
        assert not (output_dir / "INCOMPLETE").exists()

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert manifest["schema_version"] == sim.DATASET_SCHEMA_VERSION
        assert manifest["purpose"] == "off_the_shelf_model_evaluation"
        assert manifest["counts"]["frames"] == 3
        assert manifest["counts"]["scenes"] == 1
        assert manifest["camera"]["width"] == 160
        assert manifest["camera"]["height"] == 90
        assert manifest["semantic_labels"]["classes"][3] == {
            "id": 3,
            "name": "stop_sign",
        }
        assert manifest["yolo_labels"]["classes"] == [
            {"id": 0, "name": "stop_sign"}
        ]
        assert (
            manifest["yolo_labels"]["common_pretrained_yolo_class_mapping"][
                "stop_sign"
            ]
            == 11
        )

        lines = (output_dir / "frames.jsonl").read_text().splitlines()
        assert len(lines) == 3
        records = [json.loads(line) for line in lines]
        assert [record["dataset_frame_index"] for record in records] == [0, 1, 2]
        assert [record["scene_frame_index"] for record in records] == [0, 1, 2]
        assert records[0]["simulation_time_s"] == 0.0
        assert abs(records[1]["simulation_time_s"] - 0.05) < 1e-9
        assert abs(records[2]["simulation_time_s"] - 0.10) < 1e-9

        for record in records:
            with Image.open(output_dir / record["paths"]["image"]) as source:
                image = np.asarray(source.convert("RGB"))[:, :, ::-1]
            with Image.open(output_dir / record["paths"]["semantic"]) as source:
                semantic = np.asarray(source)
            instances = np.load(output_dir / record["paths"]["instance"])
            assert image is not None and image.shape == (90, 160, 3)
            assert semantic is not None and semantic.shape == (90, 160)
            assert semantic.dtype == np.uint8
            assert set(np.unique(semantic)).issubset({0, 1, 2, 3, 4})
            assert instances.shape == (90, 160)
            assert instances.dtype == np.uint32

            yolo_path = output_dir / record["paths"]["yolo_stop_sign"]
            assert yolo_path.is_file()
            for annotation in yolo_path.read_text().splitlines():
                values = annotation.split()
                assert values[0] == "0"
                assert len(values) == 5
                assert all(0.0 <= float(value) <= 1.0 for value in values[1:])

        scene_path = output_dir / manifest["scenes"][0]["path"]
        replayed_scene = sim.Scene.load(str(scene_path))
        assert replayed_scene.seed == 42
        assert replayed_scene.camera.width == 160
        assert replayed_scene.camera.height == 90

        try:
            sim.export_evaluation_dataset(config)
        except FileExistsError:
            pass
        else:
            raise AssertionError("exporter overwrote an existing dataset")


if __name__ == "__main__":
    main()
