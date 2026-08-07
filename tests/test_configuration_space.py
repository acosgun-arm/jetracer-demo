"""Heading-layer configuration-space geometry regression tests."""

from __future__ import annotations

from math import isclose, pi, sin, sqrt

import numpy as np

import jetracer_sim as sim


def _exact_model() -> sim.HeadingLayerConfigurationSpace:
    vehicle = sim.VehicleFootprintGeometry(
        wheelbase_m=0.20,
        body_width_m=0.19,
        front_overhang_m=0.055,
        rear_overhang_m=0.045,
        maximum_steering_rad=0.52,
    )
    config = sim.HeadingLayerCspaceConfig(
        layer_count=9,
        minimum_heading_rad=-15.0 * pi / 180.0,
        maximum_heading_rad=15.0 * pi / 180.0,
        grid_resolution_m=0.0025,
        obstacle_safety_margin_m=0.0,
        road_safety_margin_m=0.0,
        angular_discretization_padding_enabled=False,
        grid_cell_padding_enabled=False,
        transition_steering_fraction=0.5,
        transition_sample_spacing_m=0.0025,
    )
    return sim.HeadingLayerConfigurationSpace(
        vehicle=vehicle,
        cylinder_radius_m=0.03,
        road_width_m=0.55,
        config=config,
    )


def test_configured_nine_layers_are_symmetric() -> None:
    suite = sim.load_driving_benchmark_configuration()
    model = sim.HeadingLayerConfigurationSpace.from_driving_configuration(
        suite
    )
    assert len(model.layers) == 9
    assert isclose(model.headings_rad[0] * 180.0 / pi, -55.0)
    assert isclose(model.headings_rad[-1] * 180.0 / pi, 55.0)
    assert isclose(model.headings_rad[4], 0.0, abs_tol=1e-15)
    for left, right in zip(model.layers, reversed(model.layers)):
        assert isclose(left.heading_rad, -right.heading_rad, abs_tol=1e-15)
        assert isclose(
            left.obstacle_lateral_half_extent_m,
            right.obstacle_lateral_half_extent_m,
        )
        assert isclose(
            left.road_centre_offset_limit_m,
            right.road_centre_offset_limit_m,
        )

    expected_padding = 2.0 * model.vehicle.half_diagonal_m * sin(
        0.25 * model.config.heading_spacing_rad
    )
    assert isclose(model.angular_discretization_padding_m, expected_padding)
    # The measured Waveshare road cannot contain the complete car footprint
    # beside the inflated cylinder. The relaxed-road planner may cross a lane,
    # but aligned headings must remain less restrictive than angled headings.
    assert model.layers[4].one_side_passage_margin_m < 0.0
    assert (
        model.layers[4].one_side_passage_margin_m
        > model.layers[0].one_side_passage_margin_m
    )


def test_aligned_layer_matches_exact_rounded_rectangle() -> None:
    model = _exact_model()
    zero_layer_index = 4
    points = np.asarray(
        (
            (0.179999, 0.0),
            (0.180001, 0.0),
            (0.0, 0.124999),
            (0.0, 0.125001),
            (0.15 + 0.029999 / sqrt(2.0), 0.095 + 0.029999 / sqrt(2.0)),
            (0.15 + 0.031 / sqrt(2.0), 0.095 + 0.031 / sqrt(2.0)),
        )
    )
    occupied = model.obstacle_collision_contains(zero_layer_index, points)
    assert occupied.tolist() == [True, False, True, False, True, False]
    layer = model.layer(zero_layer_index)
    assert isclose(layer.obstacle_longitudinal_half_extent_m, 0.18)
    assert isclose(layer.obstacle_lateral_half_extent_m, 0.125)
    assert isclose(layer.road_centre_offset_limit_m, 0.18)
    assert isclose(layer.one_side_passage_margin_m, 0.055)


def test_raster_and_margin_inflation_are_conservative() -> None:
    model = _exact_model()
    raster = model.rasterize_obstacle_layer(4)
    assert raster.occupied.shape == (
        raster.y_coordinates_m.size,
        raster.x_coordinates_m.size,
    )
    centre_y = int(np.argmin(np.abs(raster.y_coordinates_m)))
    centre_x = int(np.argmin(np.abs(raster.x_coordinates_m)))
    assert raster.occupied[centre_y, centre_x]

    inflated = sim.HeadingLayerConfigurationSpace(
        vehicle=model.vehicle,
        cylinder_radius_m=model.cylinder_radius_m,
        road_width_m=model.road_width_m,
        config=sim.HeadingLayerCspaceConfig(
            layer_count=model.config.layer_count,
            minimum_heading_rad=model.config.minimum_heading_rad,
            maximum_heading_rad=model.config.maximum_heading_rad,
            grid_resolution_m=model.config.grid_resolution_m,
            obstacle_safety_margin_m=0.01,
            road_safety_margin_m=0.01,
            angular_discretization_padding_enabled=False,
            grid_cell_padding_enabled=False,
            transition_steering_fraction=(
                model.config.transition_steering_fraction
            ),
            transition_sample_spacing_m=(
                model.config.transition_sample_spacing_m
            ),
        ),
    )
    assert (
        inflated.layer(4).obstacle_lateral_half_extent_m
        > model.layer(4).obstacle_lateral_half_extent_m
    )
    assert (
        inflated.layer(4).road_centre_offset_limit_m
        < model.layer(4).road_centre_offset_limit_m
    )


def test_adjacent_transitions_are_bicycle_valid_and_include_endpoints() -> None:
    suite = sim.load_driving_benchmark_configuration()
    model = sim.HeadingLayerConfigurationSpace.from_driving_configuration(
        suite
    )
    for start_index in range(len(model.layers) - 1):
        transition = model.transition(start_index, start_index + 1)
        poses = transition.body_centre_poses
        assert isclose(poses[0, 2], model.headings_rad[start_index])
        assert isclose(poses[-1, 2], model.headings_rad[start_index + 1])
        assert isclose(poses[0, 0], 0.0, abs_tol=1e-15)
        assert isclose(poses[0, 1], 0.0, abs_tol=1e-15)
        assert 0.0 < transition.steering_rad <= model.vehicle.maximum_steering_rad
        assert transition.arc_length_m > 0.0
        assert (
            transition.arc_length_m / (poses.shape[0] - 1)
            <= model.config.transition_sample_spacing_m + 1e-15
        )
        endpoint_points = np.asarray(
            (
                (poses[0, 0], poses[0, 1]),
                (poses[-1, 0], poses[-1, 1]),
            )
        )
        assert np.all(
            model.transition_collision_contains(transition, endpoint_points)
        )
        transition_raster = model.rasterize_transition(transition)
        assert np.any(transition_raster.occupied)

    # The relaxed-road planner may use extreme heading transitions that put
    # part of the footprint outside the painted corridor.
    assert model.transition_stays_on_road(model.transition(3, 4))
    assert not model.transition_stays_on_road(model.transition(0, 1))

    reverse = model.transition(5, 4)
    assert reverse.steering_rad < 0.0
    assert reverse.arc_length_m > 0.0


def main() -> None:
    test_configured_nine_layers_are_symmetric()
    test_aligned_layer_matches_exact_rounded_rectangle()
    test_raster_and_margin_inflation_are_conservative()
    test_adjacent_transitions_are_bicycle_valid_and_include_endpoints()


if __name__ == "__main__":
    main()
