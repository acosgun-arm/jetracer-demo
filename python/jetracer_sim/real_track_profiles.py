"""Config-driven catalog of concrete real-world tracks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


REAL_TRACK_PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RealTrackProfile:
    profile_id: str
    display_name: str
    track_id: str
    manifest_path: Path
    surface: str
    geometry_status: str
    lane_marking: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RealTrackProfileCatalog:
    path: Path
    default_profile_id: str
    profiles: dict[str, RealTrackProfile]

    def profile(self, profile_id: str | None = None) -> RealTrackProfile:
        selected = profile_id or self.default_profile_id
        try:
            return self.profiles[selected]
        except KeyError as error:
            choices = ", ".join(sorted(self.profiles))
            raise ValueError(
                f"unknown real-track profile {selected!r}; choose one of: {choices}"
            ) from error


def load_real_track_profiles(path: str | Path) -> RealTrackProfileCatalog:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(
            f"real-track profile catalog does not exist: {catalog_path}"
        )
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid real-track profile catalog: {catalog_path}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("real-track profile catalog root must be an object")
    if document.get("schema_version") != REAL_TRACK_PROFILE_SCHEMA_VERSION:
        raise ValueError("unsupported real-track profile catalog schema")
    default_profile_id = document.get("default_profile_id")
    raw_profiles = document.get("profiles")
    if not isinstance(default_profile_id, str) or not default_profile_id:
        raise ValueError("real-track default profile ID must be non-empty")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise ValueError("real-track profiles must be a non-empty object")

    profiles: dict[str, RealTrackProfile] = {}
    for profile_id, value in raw_profiles.items():
        if not isinstance(profile_id, str) or not profile_id:
            raise ValueError("real-track profile IDs must be non-empty strings")
        if not isinstance(value, dict):
            raise ValueError(f"real-track profile {profile_id!r} must be an object")
        required_strings = (
            "display_name",
            "track_id",
            "manifest",
            "surface",
            "geometry_status",
        )
        for field in required_strings:
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(
                    f"real-track profile {profile_id!r} requires {field!r}"
                )
        lane_marking = value.get("lane_marking")
        if not isinstance(lane_marking, dict):
            raise ValueError(
                f"real-track profile {profile_id!r} requires lane_marking"
            )
        manifest_path = Path(value["manifest"])
        if not manifest_path.is_absolute():
            manifest_path = catalog_path.parent / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"manifest for real-track profile {profile_id!r} does not exist: "
                f"{manifest_path}"
            )
        profiles[profile_id] = RealTrackProfile(
            profile_id=profile_id,
            display_name=value["display_name"],
            track_id=value["track_id"],
            manifest_path=manifest_path,
            surface=value["surface"],
            geometry_status=value["geometry_status"],
            lane_marking=deepcopy(lane_marking),
        )
    if default_profile_id not in profiles:
        raise ValueError("real-track default profile is not defined")
    return RealTrackProfileCatalog(
        path=catalog_path,
        default_profile_id=default_profile_id,
        profiles=profiles,
    )


def resolve_real_track_manifest(
    catalog_path: str | Path,
    profile_id: str | None = None,
) -> Path:
    return load_real_track_profiles(catalog_path).profile(profile_id).manifest_path
