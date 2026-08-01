"""Smoke tests for the native Python/NumPy interface."""

import gc
import sys
import tempfile
from pathlib import Path

import numpy as np

import jetracer_sim as sim


def main() -> None:
    assert bool(sim._native.COREML_NATIVE_AVAILABLE) == (sys.platform == "darwin")
    config = sim.SceneConfig()
    config.seed = 123
    config.obstacle_count = 2
    config.stop_sign_count = 1
    scene = sim.Scene.generate(config)

    camera = sim.CameraProfile.stress_720p_200()
    camera.width = 320
    camera.height = 180
    camera.apply_nominal_intrinsics()
    engine = sim.Simulator(scene, camera)
    frame = engine.render_now()

    assert frame.y_plane.shape == (180, 320)
    assert frame.uv_plane.shape == (90, 320)
    assert frame.semantic.shape == (180, 320)
    assert frame.instance.shape == (180, 320)
    assert frame.y_plane.dtype == np.uint8
    assert frame.semantic.dtype == np.uint8
    assert frame.instance.dtype == np.uint32
    assert not frame.y_plane.flags.writeable
    assert not frame.uv_plane.flags.writeable
    assert not frame.semantic.flags.writeable
    assert not frame.instance.flags.writeable
    assert frame.to_bgr().shape == (180, 320, 3)

    retained_view = frame.y_plane
    expected_sum = int(retained_view.sum())
    del frame
    gc.collect()
    assert int(retained_view.sum()) == expected_sum

    frames = engine.advance(sim.VehicleCommand(0.5, 0.03), 0.012)
    assert len(frames) == 2
    assert abs(frames[0].simulation_time_s - 0.005) < 1e-12
    assert abs(frames[1].simulation_time_s - 0.010) < 1e-12

    with tempfile.TemporaryDirectory(prefix="jetracer-sim-test-") as directory:
        path = Path(directory) / "scene.json"
        scene.save(str(path))
        loaded = sim.Scene.load(str(path))
        assert loaded.seed == scene.seed
        assert len(loaded.centerline) == len(scene.centerline)


if __name__ == "__main__":
    main()
