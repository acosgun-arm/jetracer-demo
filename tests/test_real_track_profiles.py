"""Concrete real-world track profile catalog tests."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jetracer_sim as sim


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG = REPOSITORY_ROOT / "configs" / "real_tracks.json"


def test_configured_profiles_resolve_independent_datasets() -> None:
    catalog = sim.load_real_track_profiles(CATALOG)
    assert catalog.default_profile_id == "waveshare"
    assert set(catalog.profiles) == {"waveshare", "floor"}
    waveshare = catalog.profile()
    floor = catalog.profile("floor")
    assert waveshare.track_id == "waveshare_3x2"
    assert waveshare.lane_marking["road_width_m"] == 0.339
    assert floor.track_id == "floor_v1"
    assert floor.geometry_status == "awaiting_design_and_measurement"
    assert waveshare.manifest_path != floor.manifest_path
    assert sim.load_real_track_dataset(floor.manifest_path).document["track"][
        "track_id"
    ] == "floor_v1"


def test_new_profile_requires_no_python_change() -> None:
    with TemporaryDirectory(prefix="jetracer-track-profile-") as directory:
        root = Path(directory)
        manifest = root / "manifest.json"
        manifest.write_text(
            (REPOSITORY_ROOT / "datasets/real_tracks/floor/manifest.json")
            .read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        document = {
            "schema_version": 1,
            "default_profile_id": "test",
            "profiles": {
                "test": {
                    "display_name": "Test track",
                    "track_id": "test_v1",
                    "manifest": "manifest.json",
                    "surface": "test_surface",
                    "geometry_status": "measured",
                    "lane_marking": {"boundary_colour": "blue"},
                }
            },
        }
        path = root / "tracks.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        assert sim.load_real_track_profiles(path).profile().track_id == "test_v1"


def test_unknown_profile_is_rejected_with_available_choices() -> None:
    catalog = sim.load_real_track_profiles(CATALOG)
    try:
        catalog.profile("missing")
    except ValueError as error:
        assert "floor" in str(error) and "waveshare" in str(error)
    else:
        raise AssertionError("unknown track profile was accepted")
