"""Source/install configuration-resource resolution tests."""

from __future__ import annotations

import jetracer_sim as sim
from jetracer_sim.resource_paths import configuration_resource


def test_default_configuration_resources_exist() -> None:
    defaults = (
        sim.DEFAULT_RUNTIME_CONFIG_PATH,
        sim.DEFAULT_NATIVE_SIMULATOR_CONFIG_PATH,
        sim.DEFAULT_DRIVING_BENCHMARK_CONFIG_PATH,
        sim.DEFAULT_PLATFORM_CONFIG_PATH,
        sim.DEFAULT_CAMERA_PROFILE_PATH,
        sim.DEFAULT_ACTUATOR_PROFILE_PATH,
        sim.DEFAULT_STATE_PROFILE_PATH,
        sim.DEFAULT_DEPLOYMENT_POLICY_PATH,
        sim.DEFAULT_PREFLIGHT_CONFIG_PATH,
        sim.DEFAULT_BRINGUP_PLAN_PATH,
    )
    assert all(path.is_file() for path in defaults)


def test_unsafe_configuration_resource_is_rejected() -> None:
    for path in ("../outside.json", "/absolute.json"):
        try:
            configuration_resource(path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe configuration resource accepted: {path}")


def main() -> None:
    test_default_configuration_resources_exist()
    test_unsafe_configuration_resource_is_rejected()


if __name__ == "__main__":
    main()
