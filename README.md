# JetRacer high-rate simulator

[![Headless CI](https://github.com/acosgun-arm/jetracer-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/acosgun-arm/jetracer-demo/actions/workflows/ci.yml)

A lightweight simulator and control stack for developing JetRacer vision
software before the hardware arrives. It provides a C++ CPU renderer, a
kinematic bicycle model, configurable wide-angle cameras, lane/road semantics,
stop signs and obstacles, Python perception adapters, and a latency-aware speed
governor.

The basic demo requires no downloaded model weights: it uses simulator semantic
labels with selectable inference delays so model switching, frame replacement,
steering, telemetry, and adaptive speed can be exercised end to end.

## Requirements

The primary development platform is macOS with Homebrew:

```bash
brew install cmake ninja opencv pybind11 python
```

Python 3.10 or newer and a C++20 compiler are required.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Re-run the final command after adding or updating package modules. Optional
off-the-shelf model dependencies can be installed later with:

```bash
python -m pip install -e '.[pretrained,onnx]'
```

## Run the basic simulator demo

Start the simulated camera, lane controller, model switcher, adaptive speed
governor, telemetry logger, and local browser dashboard:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json \
  --model-config configs/demo_models.json \
  --benchmark-registry benchmarks/demo_model_benchmarks.json
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The server binds only to
localhost and does not open a desktop application automatically.

Controls:

- `1`–`9`: select a model
- `[` / `]`: decrease or increase requested speed
- `P`: pause
- `L`: toggle segmentation and detection overlays
- `R`: reset the controllers
- `Space`: stop
- `Q`: quit

Telemetry is written to a unique JSONL file under `build/telemetry/`. Use
`--no-log` to disable it or `--log PATH` to choose the output.

### Headless smoke test

This exercises the same control loop without creating a browser or native
window:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json \
  --model-config configs/demo_models.json \
  --benchmark-registry benchmarks/demo_model_benchmarks.json \
  --headless --duration 3 --switch-every 0.75 --no-log
```

### Closed-loop and benchmark runs

Run one deterministic vision-controlled lap:

```bash
python examples/closed_loop_lap.py
```

Run the configured lane, stop-sign, and pedestrian-avoidance benchmark suite:

```bash
python tools/run_driving_benchmarks.py --scenario full
```

Use `--scenario lane|stops|pedestrian`, `--track TRACK_ID`, or `--laps N`
for focused runs. Results are written under `build/benchmarks/` and include
off-road events, centreline deviation, average speed, stop compliance, and
collisions.

Export a headless 10-second Waveshare-track clip at the ELP camera's native
1920x1200 cadence, together with lossless semantic ground truth and metadata:

```bash
python tools/export_synthetic_clip.py
```

Outputs are written under `benchmarks/synthetic_clips/`. Use `--track`,
`--duration`, `--speed`, or `--profile` to override the configured defaults.
Evaluate the configured off-the-shelf segmenters against an exported clip:

```bash
python tools/evaluate_synthetic_clip.py benchmarks/synthetic_clips/CLIP_DIRECTORY
```

## Build and test

For the complete native and Python test suite:

```bash
cmake -S . -B build-python -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DJETRACER_SIM_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-python
ctest --test-dir build-python --output-on-failure
```

A native-only build is available with
`-DJETRACER_SIM_BUILD_PYTHON=OFF`.

## Optional off-the-shelf models

The initial deployment manifest supports SegFormer road segmentation and
YOLO11n COCO detection, with CPU, Core ML, and Jetson TensorRT variants. Model
weights are intentionally not committed. Follow [models/README.md](models/README.md)
to prepare and validate artifacts.

Once the required artifacts exist:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json \
  --model-config configs/off_the_shelf_models.json \
  --detector-config configs/off_the_shelf_models.json
```

The demo automatically prefers Core ML/Neural Engine FP16 on macOS and
TensorRT FP16 on NVIDIA, with CPU as a fallback. Use `--model KEY` to override
the selected variant.

Unavailable variants are reported and skipped. The simulator-delay models above
remain the fastest way to exercise the complete application without external
downloads.

## Simulator/real-hardware switch

The control process selects all platform I/O through one JSON file:

```bash
python tools/check_platform.py --platform configs/platforms/sim.json
python tools/check_platform.py --platform configs/platforms/jetracer-pro.json
```

Use `configs/platforms/sim.json` for the native simulator. The JetRacer Pro
profile is deliberately fail-closed: motor output remains disabled until the
camera, vehicle geometry, actuator limits, state source, watchdog, and explicit
arming procedure have been validated on the delivered hardware. Controller and
perception code do not require simulator-specific branches.

See:

- [Hardware bring-up](docs/HARDWARE_BRINGUP.md)
- [Jetson setup](docs/JETSON_SETUP.md)
- [Deployment and rollback](docs/DEPLOYMENT_RECOVERY.md)

## GUI and camera safety on macOS

The browser viewer is the default. Do not use `--open-browser` while the Mac
may be locked.

The native OpenCV viewer is double opt-in:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json \
  --model-config configs/demo_models.json \
  --benchmark-registry benchmarks/demo_model_benchmarks.json \
  --viewer opencv --allow-native-gui
```

Use it only during an unlocked interactive session; macOS can abort a process
that creates a native GUI while the screen is locked. Headless runs, the local
browser server, and the AVFoundation camera characterization tool do not create
an OpenCV window.

For the ELP camera, request permission once while unlocked and then inspect or
measure it headlessly:

```bash
python tools/characterize_camera.py list --request-permission
python tools/characterize_camera.py list
python tools/characterize_camera.py measure
```

## Configuration

Runtime and experiment values are versioned rather than embedded in control
logic:

- `configs/platforms/*.json`: simulator or real camera/actuator/state drivers
- `configs/runtime_defaults.json`: controller, governor, inference, and UI settings
- `configs/driving_benchmarks.json`: tracks, vehicle geometry, and scenarios
- `configs/native_simulator_defaults.json`: renderer and camera defaults
- `configs/off_the_shelf_models.json`: deployable model variants
- `benchmarks/*.json`: measured or synthetic model performance

The governor's `baseline_distance_per_frame_m` is the allowed travel per
processed vision frame. Change it only from measured real-vehicle control and
stopping performance.
