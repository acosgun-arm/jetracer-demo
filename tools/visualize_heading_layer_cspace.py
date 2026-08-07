#!/usr/bin/env python3
"""Render headless diagnostics for heading-layer configuration-space geometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import degrees
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, Patch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

import jetracer_sim as sim  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cspace-config", type=Path)
    parser.add_argument("--track", default="waveshare_3x2")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("build/visualizations/heading-layer-cspace"),
    )
    return parser.parse_args()


def _draw_occupancy(
    axis: Any,
    raster: sim.RasterizedCspace,
    *,
    colour: str,
    alpha: float,
) -> None:
    occupancy = np.ma.masked_where(~raster.occupied, raster.occupied)
    axis.imshow(
        occupancy,
        origin="lower",
        interpolation="nearest",
        extent=(
            raster.x_coordinates_m[0],
            raster.x_coordinates_m[-1],
            raster.y_coordinates_m[0],
            raster.y_coordinates_m[-1],
        ),
        cmap=ListedColormap((colour,)),
        alpha=alpha,
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )


def _render_layers(
    model: sim.HeadingLayerConfigurationSpace, output_path: Path
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(13.0, 10.0), sharex=True, sharey=True)
    road_half_width = 0.5 * model.road_width_m
    maximum_x = max(
        layer.obstacle_longitudinal_half_extent_m for layer in model.layers
    ) + model.config.grid_resolution_m

    for axis, layer in zip(axes.flat, model.layers):
        raster = model.rasterize_obstacle_layer(layer.index)
        limit = max(layer.road_centre_offset_limit_m, 0.0)
        if limit > 0.0:
            axis.axhspan(
                -limit,
                limit,
                color="#2ca02c",
                alpha=0.13,
                zorder=0,
            )
        axis.axhspan(
            road_half_width,
            road_half_width + model.config.grid_resolution_m,
            color="#333333",
            alpha=0.8,
            zorder=3,
        )
        axis.axhspan(
            -road_half_width - model.config.grid_resolution_m,
            -road_half_width,
            color="#333333",
            alpha=0.8,
            zorder=3,
        )
        _draw_occupancy(axis, raster, colour="#d62728", alpha=0.38)
        axis.add_patch(
            Circle(
                (0.0, 0.0),
                model.cylinder_radius_m,
                facecolor="#9467bd",
                edgecolor="#5b2c83",
                linewidth=1.0,
                zorder=4,
            )
        )
        axis.axhline(0.0, color="#777777", linewidth=0.6, zorder=1)
        axis.axvline(0.0, color="#777777", linewidth=0.6, zorder=1)
        passage_mm = 1000.0 * layer.one_side_passage_margin_m
        passage_colour = "#20732c" if passage_mm >= 0.0 else "#a11d1d"
        axis.set_title(f"{degrees(layer.heading_rad):+.2f}°")
        axis.text(
            0.5,
            0.02,
            f"one-side corridor {passage_mm:+.1f} mm",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            color=passage_colour,
            fontsize=8,
        )
        axis.set_xlim(-maximum_x, maximum_x)
        axis.set_ylim(-road_half_width - 0.015, road_half_width + 0.015)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#b0b0b0", linewidth=0.35, alpha=0.35)

    figure.suptitle(
        "Waveshare track: nine heading-layer configuration spaces",
        fontsize=15,
        y=0.99,
    )
    figure.supxlabel(
        "vehicle body-centre longitudinal position relative to cylinder (m)",
        y=0.01,
    )
    figure.supylabel("vehicle body-centre lateral position relative to cylinder (m)")
    figure.legend(
        handles=(
            Patch(facecolor="#d62728", alpha=0.38, label="collision C-space"),
            Patch(facecolor="#2ca02c", alpha=0.13, label="road-admissible centre band"),
            Patch(facecolor="#9467bd", label="30 mm cylinder"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    figure.tight_layout(rect=(0.03, 0.045, 1.0, 0.90), h_pad=1.8)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _render_transitions(
    model: sim.HeadingLayerConfigurationSpace, output_path: Path
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(15.0, 8.5))
    for axis, start_index in zip(axes.flat, range(len(model.layers) - 1)):
        transition = model.transition(start_index, start_index + 1)
        raster = model.rasterize_transition(transition)
        footprints = model.transition_footprints_m(transition)
        _draw_occupancy(axis, raster, colour="#d62728", alpha=0.32)
        sample_indices = np.unique(
            np.linspace(0, len(footprints) - 1, 5).astype(int)
        )
        for footprint_index in sample_indices:
            footprint = footprints[footprint_index]
            is_endpoint = footprint_index in {0, len(footprints) - 1}
            axis.add_patch(
                Polygon(
                    footprint,
                    closed=True,
                    facecolor="#1f77b4",
                    edgecolor="#174a73",
                    linewidth=1.1 if is_endpoint else 0.5,
                    alpha=0.28 if is_endpoint else 0.10,
                    zorder=3,
                )
            )
        poses = transition.body_centre_poses
        axis.plot(
            poses[:, 0],
            poses[:, 1],
            color="#111111",
            linewidth=1.2,
            marker="o",
            markevery=(0, len(poses) - 1),
            markersize=3.0,
            zorder=4,
        )
        start_degrees = degrees(model.headings_rad[start_index])
        end_degrees = degrees(model.headings_rad[start_index + 1])
        axis.set_title(
            f"{start_degrees:+.2f}° → {end_degrees:+.2f}°\n"
            f"arc {1000.0 * transition.arc_length_m:.1f} mm"
        )
        x_padding = model.effective_obstacle_radius_m
        y_padding = model.effective_obstacle_radius_m
        axis.set_xlim(
            raster.x_coordinates_m[0] - x_padding,
            raster.x_coordinates_m[-1] + x_padding,
        )
        axis.set_ylim(
            raster.y_coordinates_m[0] - y_padding,
            raster.y_coordinates_m[-1] + y_padding,
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#b0b0b0", linewidth=0.35, alpha=0.35)

    figure.suptitle(
        "Adjacent bicycle-model transitions and their swept collision regions",
        fontsize=15,
        y=0.99,
    )
    figure.supxlabel("local longitudinal position (m)", y=0.01)
    figure.supylabel("local lateral position (m)")
    figure.legend(
        handles=(
            Patch(facecolor="#d62728", alpha=0.32, label="swept collision C-space"),
            Patch(facecolor="#1f77b4", alpha=0.28, label="sampled vehicle footprint"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.03, 0.045, 1.0, 0.88), h_pad=2.2)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_summary(
    model: sim.HeadingLayerConfigurationSpace,
    *,
    track_id: str,
    output_path: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "track_id": track_id,
        "planning_reference": "vehicle_body_centre",
        "vehicle": asdict(model.vehicle),
        "cylinder_radius_m": model.cylinder_radius_m,
        "road_width_m": model.road_width_m,
        "config": asdict(model.config),
        "angular_discretization_padding_m": (
            model.angular_discretization_padding_m
        ),
        "grid_cell_padding_m": model.grid_cell_padding_m,
        "effective_obstacle_radius_m": model.effective_obstacle_radius_m,
        "layers": [
            {
                **asdict(layer),
                "heading_degrees": degrees(layer.heading_rad),
            }
            for layer in model.layers
        ],
        "transitions": [
            {
                "start_layer_index": transition.start_layer_index,
                "end_layer_index": transition.end_layer_index,
                "steering_rad": transition.steering_rad,
                "arc_length_m": transition.arc_length_m,
                "sample_count": int(transition.body_centre_poses.shape[0]),
                "stays_on_road_from_centre": model.transition_stays_on_road(
                    transition
                ),
            }
            for transition in (
                model.transition(index, index + 1)
                for index in range(len(model.layers) - 1)
            )
        ],
    }
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    arguments = parse_arguments()
    suite = sim.load_driving_benchmark_configuration(arguments.config)
    model = sim.HeadingLayerConfigurationSpace.from_driving_configuration(
        suite,
        track_id=arguments.track,
        cspace_configuration=sim.load_heading_layer_cspace_configuration(
            arguments.cspace_config
        ),
    )
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    layers_path = output_directory / "heading-layer-obstacle-cspace.png"
    transitions_path = output_directory / "heading-layer-transitions.png"
    summary_path = output_directory / "heading-layer-cspace.json"
    _render_layers(model, layers_path)
    _render_transitions(model, transitions_path)
    _write_summary(
        model,
        track_id=arguments.track,
        output_path=summary_path,
    )
    print(layers_path)
    print(transitions_path)
    print(summary_path)


if __name__ == "__main__":
    main()
