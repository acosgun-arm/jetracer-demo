"""Tests for configurable colour thresholding and robust boundary fitting."""

from __future__ import annotations

import numpy as np
from pathlib import Path

import jetracer_sim as sim


def config(*, birdseye: bool = False) -> sim.ColorLaneSegmentationConfig:
    points = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    return sim.ColorLaneSegmentationConfig(
        processing_width=100,
        processing_height=60,
        hsv_ranges=(sim.HsvRange((0, 0, 200), (179, 40, 255)),),
        roi_top_fraction=0.0,
        morphology_close_kernel=0,
        morphology_open_kernel=0,
        minimum_run_width_px=2,
        minimum_lane_width_px=20,
        polynomial_degree=2,
        fit_iterations=3,
        minimum_fit_points=8,
        residual_floor_px=2.0,
        residual_quantile=0.75,
        residual_multiplier=2.0,
        path_sample_count=10,
        birdseye_source_points=points if birdseye else None,
        birdseye_destination_points=points if birdseye else None,
    )


def paired_boundaries() -> np.ndarray:
    image = np.zeros((60, 100, 3), dtype=np.uint8)
    for row in range(image.shape[0]):
        offset = int(round(4.0 * (row / image.shape[0]) ** 2))
        image[row, 18 + offset : 22 + offset] = 255
        image[row, 78 + offset : 82 + offset] = 255
    return image


def main() -> None:
    adapter = sim.ColorLaneSegmentationAdapter(config())
    prediction = adapter.infer(paired_boundaries())
    assert prediction.road_class_id == 1
    assert np.all(prediction.labels[:, 30:70] == 1)
    assert np.all(prediction.labels[:, :10] == 0)
    diagnostics = adapter.latest_diagnostics
    assert diagnostics.confidence > 0.9
    assert diagnostics.observed_rows == 60
    assert len(diagnostics.center_path_xy) == 10
    assert not diagnostics.birdseye_applied

    identity_adapter = sim.ColorLaneSegmentationAdapter(config(birdseye=True))
    identity = identity_adapter.infer(paired_boundaries())
    assert np.array_equal(identity.labels, prediction.labels)
    assert identity_adapter.latest_diagnostics.birdseye_applied

    missing_boundary = paired_boundaries()
    missing_boundary[:, 75:] = 0
    failed = adapter.infer(missing_boundary)
    assert not np.any(failed.labels)
    assert adapter.latest_diagnostics.confidence == 0.0

    parsed = sim.hsv_ranges_from_config(
        [{"lower": [1, 2, 3], "upper": [4, 5, 6]}]
    )
    assert parsed == (sim.HsvRange((1, 2, 3), (4, 5, 6)),)

    profile_path = (
        Path(__file__).resolve().parents[1]
        / "configs/color_lane/waveshare-sim-white.json"
    )
    shared_config = sim.load_color_lane_profile(profile_path)
    assert shared_config.processing_width == 640
    native_adapter = sim.ColorLaneSegmentationAdapter(
        shared_config,
        native_profile_path=profile_path,
    )
    native_prediction = native_adapter.infer(paired_boundaries())
    assert native_prediction.labels.shape == paired_boundaries().shape[:2]
    assert np.any(native_prediction.labels)


if __name__ == "__main__":
    main()
