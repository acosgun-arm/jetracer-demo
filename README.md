# JetRacer high-rate simulator

[![Headless CI](https://github.com/acosgun-arm/jetracer-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/acosgun-arm/jetracer-demo/actions/workflows/ci.yml)

A lightweight simulator and control stack for developing JetRacer vision
software before the hardware arrives. It provides a C++ CPU renderer, a
kinematic bicycle model, configurable wide-angle cameras, lane/road semantics,
stop signs and obstacles, Python perception adapters, and a latency-aware speed
governor.

The master simulator profile uses the configured off-the-shelf vision models.
A dependency-free smoke command uses simulator semantic labels with selectable
inference delays.

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

For the configurable colour-lane backend without native GUI dependencies:

```bash
python -m pip install -e '.[classical-vision]'
```

## Run the basic simulator demo

Start the simulated camera, lane controller, model switcher, adaptive speed
governor, telemetry logger, and local browser dashboard:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). The server binds only to
localhost and does not open a desktop application automatically.

Controls:

- Dashboard selectors: lane-only/hazards mode, vision model, configured lateral
  controller, and centerline/local/minimum-time racing line
- `1`–`9`: select a vision model from the keyboard
- `[` / `]`: decrease or increase requested speed
- `P`: pause
- `L`: toggle segmentation and detection overlays
- `R`: reset the controllers
- `Space`: stop
- `Q`: quit

Each selection updates the exact speed-certification lookup. A real platform
remains stopped when the selected combination has no current certification.
Lane-only mode pauses object detection; hazards mode enables the configured
detector and stop controller. Both modes keep segmentation lane following
active. Set the startup mode with `--driving-mode lane-only|hazards`.

Telemetry is written to a unique JSONL file under `build/telemetry/`. Use
`--no-log` to disable it or `--log PATH` to choose the output.

### Preview and capture the ELP camera on macOS

With the USB camera connected, start the motor-disabled Mac profile:

```bash
.venv/bin/python examples/realtime_demo.py \
  --platform-config configs/platforms/mac-elp.json
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Use **Show raw feed** to
switch between the unmodified camera image and processed overlays. The capture
panel saves raw PNG snapshots or MP4 video under `datasets/real_track/media/`
and registers each result in the dataset manifest. Select the split, lighting,
track section and scene before capturing. Video queue, written-frame and
dropped-frame counts remain visible while recording.

Grant camera permission while the Mac is unlocked the first time. This command
does not create an OpenCV/native window and does not enable vehicle motors.
It also fails closed if the configured ELP identity is absent or if macOS
negotiates a camera mode below the profile's acceptance threshold.

### Calibrate lane colours from video

After recording a real-track MP4, extract diverse frames without opening a
native window. Supplying the current profile also prioritizes its uncertain
and failure cases:

```bash
.venv/bin/python tools/prepare_video_lane_calibration.py \
  datasets/real_track/media/VIDEO.mp4 build/lane-calibration/waveshare-elp \
  --track-profile waveshare --camera-profile elp_112 \
  --current-profile configs/color_lane/waveshare-sim-white.json
```

Annotate representative lane/background pixels and optional boundary
polylines at [http://127.0.0.1:8766](http://127.0.0.1:8766):

```bash
.venv/bin/python tools/annotate_lane_video.py \
  build/lane-calibration/waveshare-elp
```

After annotating one reliable frame, generate forward optical-flow proposals;
the browser marks them as proposals and never overwrites reviewed work:

```bash
.venv/bin/python tools/propagate_lane_annotations.py \
  build/lane-calibration/waveshare-elp
```

Use **Uncertain first** to prioritize low-confidence, abrupt, poorly lit,
blurred, or weakly propagated frames. Draw a **Road polygon** on a small
validation subset, then measure the generated road masks:

```bash
.venv/bin/python tools/benchmark_video_lane_masks.py \
  build/lane-calibration/waveshare-elp \
  configs/color_lane/waveshare/elp-112.json
```

Fit an HSV range, report Lab bounds, and export a profile compatible with the
Python and native implementations:

```bash
.venv/bin/python tools/calibrate_video_lane.py \
  build/lane-calibration/waveshare-elp \
  --profile-id waveshare-elp-112 \
  --output configs/color_lane/waveshare/elp-112.json
```

Recordings are kept immutable. Keyframes are selected using illumination,
colour, sharpness, scene novelty, and optional current-detector confidence;
annotations are stored as normalized coordinates in `annotations.json`.

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

Run the deterministic single-cylinder avoidance benchmark on every track:

```bash
.venv/bin/python tools/run_driving_benchmarks.py \
  --scenario cylinder --avoidance-method clearance-aware --track all
```

The simulated/real reference cylinder dimensions are configured under
`objects.cylinder` in `configs/driving_benchmarks.json`; visual `radius_m`,
physical `collision_radius_m`, and `height_m` are independent.

Run the reproducible cylinder robustness suites (headless):

```bash
.venv/bin/python tools/run_cylinder_robustness.py \
  --mode placement-grid --track all --exploratory
.venv/bin/python tools/run_cylinder_robustness.py \
  --mode joint-monte-carlo --track all --exploratory
```

The first command exhaustively checks the configured longitudinal/lateral
placement grid. The second jointly samples cylinder dimensions, speed, and
perception latency, dropout, and bias. Remove `--exploratory` to return a
failing exit status when any collision or off-road event occurs.

Run the one-to-three-cylinder DWA matrix, optionally with deterministic
perception faults:

```bash
.venv/bin/python tools/run_multi_obstacle_benchmarks.py \
  --object-counts 1 2 3 --controllers adaptive_with_avoidance_pursuit \
  --planners dynamic_window --layout-mode random \
  --segmentation-noise moderate \
  --obstacle-noise moderate --path-filter temporal \
  --output build/benchmarks/multi-obstacle.json
```

Unsafe cases produce a non-zero exit status; add `--exploratory` while tuning.
Render its compact CI-style table with
`tools/summarize_multi_obstacle_benchmark.py REPORT.json`.

Render a compact headless top-down MP4 for one exact placement-grid case:

```bash
.venv/bin/python -m pip install -e '.[export]'
brew install ffmpeg
.venv/bin/python tools/render_top_down_scenario.py --case 1
```

The video distinguishes perceived centre points, the display-smoothed control
reference, the selected look-ahead goal, and a constant-command bicycle rollout.
It uses the configured winning controller, configured obstacle planner, and
temporal path filter by default; override them with `--controller`,
`--local-planner`, and `--path-filter`.

Use `--scenario lane|stops|pedestrian|cylinder`, or render all 49 configured
cylinder placements with `--all-placement-cases`. Video size, frame rate,
encoding and drawing defaults are under `top_down_video` in
`configs/runtime_defaults.json`; CLI overrides include `--width`, `--fps`,
`--crf`, and `--output`.

Compare centreline, smoothing-based, and minimum-time racing lines headlessly:

```bash
.venv/bin/python tools/run_control_benchmarks.py \
  --methods pure_pursuit \
  --path-planners centerline local-racing-line minimum-time-racing-line \
  --perception oracle --track all
```

Isolate path-planner effects from live inference-speed drift with paired,
alternating-order Core ML trials:

```bash
.venv/bin/python tools/run_paired_path_benchmarks.py \
  --model-key 4 --trials 2 --fixed-governor-fps 90
```

Add curvature-aware speed planning and request a high cruise speed:

```bash
.venv/bin/python tools/run_control_benchmarks.py \
  --methods pure_pursuit \
  --path-planners centerline minimum-time-racing-line \
  --speed-planners curvature --speed 2.5 \
  --perception oracle --track all
```

Certify the maximum safe speed for one complete vision/control combination
and update `benchmarks/certified_speed_limits.json`:

```bash
.venv/bin/python tools/find_max_safe_speed.py \
  --perception actual --model-key 4 \
  --method pure_pursuit --path-filter off \
  --path-planner minimum-time-racing-line --speed-planner curvature
```

The search uses the configured tracks, laps, trials, safety gates, and
simulation-to-real speed factor from `configs/driving_benchmarks.json`.
Temporal path filtering remains available as the opt-in
`--path-filter temporal` variant for noisy real-world perception.
Simulator runtime enforcement is optional; the JetRacer platform requires an
exact, non-stale registry match before autonomous motion.

Preview every currently runnable vision model × configured controller × racing
line combination without running benchmarks:

```bash
.venv/bin/python tools/certify_speed_matrix.py --dry-run
```

Run the matrix sequentially and resume safely after interruption:

```bash
.venv/bin/python tools/certify_speed_matrix.py \
  --output-dir build/benchmarks/full-speed-matrix --resume
```

Unavailable models are recorded and excluded. Existing exact certifications
are skipped unless `--rerun-certified` is supplied. Use `--model-keys`,
`--methods`, and `--path-planners` to run a smaller matrix.

Promote a completed matrix for the dashboard and verify that every configured
model, controller, racing line, and track has current coverage:

```bash
.venv/bin/python tools/promote_speed_certification_matrix.py \
  --summary build/benchmarks/full-speed-matrix/summary.json
.venv/bin/python tools/check_speed_certification_coverage.py
```

The browser dashboard displays the promoted results as a speed-certification
heatmap with per-track candidate metrics. The full matrix is a manual or
scheduled job for a suitable self-hosted runner; it is not run on every push.

Run the push-time Waveshare regression gate (color thresholding, adaptive pure
pursuit, and DWA obstacle avoidance) with:

```bash
.venv/bin/python tools/run_deployment_stack_benchmark.py --overwrite
```

This compact gate compares safety, completion count, and mean speed with
`benchmarks/deployment_stack_benchmark_baseline.json`. It is a regression
smoke test, not full deployment certification.

Run the same benchmark with the master platform configuration's actual
segmentation and detection models, latest-frame workers, and speed governor:

```bash
python tools/run_driving_benchmarks.py \
  --platform configs/platforms/sim.json \
  --perception actual --scenario lane --track waveshare_3x2 --laps 1
```

Use `--scenario lane|stops|pedestrian`, `--track TRACK_ID`, or `--laps N`
for focused runs. Results are written under `build/benchmarks/` and include
off-road events, centreline deviation, average speed, stop compliance, and
collisions. Lane, stop-sign, and pedestrian scenarios enforce configured
acceptance thresholds and return a failing exit status on regressions; use
`--no-enforce-acceptance` only for exploratory runs.
Actual-perception lane scenarios disable the object detector automatically;
stop-sign and pedestrian scenarios keep it active.
Actual stop-sign runs load `configs/stop_sign_latency_profiles.json`. The
controller uses its P99 detector and actuator budget plus live result age; an
unprofiled detector is held at zero speed. The same profile rate-limits detector
submissions so its configured cadence is included in the braking budget.

Preview or run the shortened stop-sign speed certification:

```bash
.venv/bin/python tools/certify_stop_sign_braking.py --dry-run
.venv/bin/python tools/certify_stop_sign_braking.py
```

It starts from the analytical braking/latency limit, pre-benchmarked vision FPS,
and the current lane-speed certificate. It screens 90%, 100%, and 110% with one
lap, then runs longer confirmation only for the highest eligible survivor.
Search parameters and the simulation-to-real margin are in
`configs/stop_sign_benchmark.json`.

Compare the configured lateral controllers on identical oracle road masks:

```bash
python tools/run_control_benchmarks.py \
  --platform configs/platforms/sim.json
```

Add real Core ML perception or deterministic latency profiles with:

```bash
python tools/run_control_benchmarks.py --perception actual --model-key 3 4 \
  --track waveshare_3x2 --laps 1
python tools/run_control_benchmarks.py --perception actual --model-key 3 \
  --path-filters off temporal --track waveshare_3x2 --laps 1
python tools/run_control_benchmarks.py --perception simulated-latency \
  --model-config configs/demo_models.json --model-key 1 2 3 4 \
  --track waveshare_3x2 --laps 1
```

Add `--profile-stages` to include per-stage mean, P50, P95, P99, and maximum
closed-loop latencies in each benchmark result.

Measure sensitivity to an incorrect camera height or pitch calibration:

```bash
python tools/run_camera_mount_benchmarks.py \
  --platform configs/platforms/sim.json
```

The renderer uses each configured test mount while ground projection retains
the nominal mount, reproducing calibration error rather than a matched virtual
camera move.

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

Real-track images and high-rate videos use a separate, train-free calibration
and held-out evaluation pipeline. The empty manifest is ready now; see
[`docs/REAL_TRACK_DATASET.md`](docs/REAL_TRACK_DATASET.md).

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
Headless CI also archives its one-lap oracle acceptance report as the
`oracle-driving-acceptance` artifact and publishes its core metrics in the job
summary. It compares deterministic driving metrics with the versioned oracle
baseline and fails when configured regression tolerances are exceeded.

After reviewing an intentional behavior change, refresh that baseline with
`tools/promote_driving_benchmark_baseline.py`; replacement requires the
explicit `--force` option, the regression policy passed through `--config`, and
a report whose acceptance gates all passed.

## Optional off-the-shelf models

The platform selects Cityscapes SegFormer-B0/B1 road segmentation from
`configs/road_segmentation_models.json` and YOLO11n COCO detection from
`configs/off_the_shelf_models.json`. CPU, Core ML, and Jetson TensorRT variants
are defined; model weights are intentionally not committed. Follow
[models/README.md](models/README.md) to prepare and validate artifacts.

Once the required artifacts exist:

```bash
python examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json
```

The demo automatically prefers Core ML/Neural Engine FP16 on macOS and
TensorRT FP16 on NVIDIA, with CPU as a fallback. Use `--model KEY` to override
the selected variant. On macOS, key `4` is the balanced 384-pixel default and
key `3` is the high-rate 256-pixel option.

Record and gate the macOS 200 Hz/Core ML path headlessly (no GUI is opened):

```bash
.venv/bin/python tools/run_realtime_performance_gate.py
```

The thresholds, warmup, expected model, and source-rate tolerance are in
`configs/realtime_performance_regression.json`. The gate derives source FPS
from capture counters; `measured_camera_fps` is only the consumer loop rate.
CI also checks `benchmarks/realtime_performance_evidence.json` and fails when
performance-relevant model or runtime sources have changed since that passing
Mac measurement. Evaluate an existing telemetry file with
`tools/check_realtime_performance.py` when a fresh run is unnecessary.
After reviewing a fresh passing report, update CI evidence explicitly:

```bash
.venv/bin/python tools/promote_realtime_performance_evidence.py \
  build/performance-gate/report-YYYYMMDDTHHMMSSZ.json --force
```

Unavailable variants are reported and skipped. The simulator-delay models above
remain the fastest way to exercise the complete application without external
downloads.

## Simulator/real-hardware switch

The control process selects all platform I/O through one JSON file:

```bash
python tools/check_platform.py --platform configs/platforms/sim.json
python tools/check_platform.py --platform configs/platforms/jetracer-pro.json
```

Use `configs/platforms/sim.json` for the native simulator and
`configs/platforms/mac-elp.json` for motor-disabled ELP preview/capture on a
Mac. The JetRacer Pro
profile is deliberately fail-closed: motor output remains disabled until the
camera, vehicle geometry, actuator limits, state source, watchdog, and explicit
arming procedure have been validated on the delivered hardware. Controller and
perception code do not require simulator-specific branches.

Physical camera definitions are separate from the vehicle platform:

- `configs/cameras/elp-112.json`: ELP macOS/Jetson modes at 120 or 200 Hz
- `configs/cameras/imx219-160.json`: original JetRacer IMX219 Jetson modes

Select a camera in the platform configuration with only:

```json
"camera": {
  "profile_config": "../cameras/imx219-160.json",
  "mode_id": "jetson_720p_60"
}
```

Use `elp-112.json` with `jetson_1200p_120` or `jetson_720p_200` for the ELP.
Adding another camera requires a new runtime-profile JSON plus its physical
identity/calibration entry in `configs/hardware/cameras.json`; controller code
does not change. Camera-profile content is included in benchmark fingerprints,
so changing a camera or mode invalidates stale speed certification.

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
- `configs/stop_sign_benchmark.json`: shortened analytical stop-speed search
- `configs/stop_sign_latency_profiles.json`: measured stop-detector latency,
  ranging margin, and actuation budgets
- `configs/native_simulator_defaults.json`: renderer and camera defaults
- `configs/road_segmentation_models.json`: neural and colour-based road variants
- `configs/off_the_shelf_models.json`: detector and legacy ADE20K variants
- `benchmarks/road_model_benchmarks.json`: measured road-model performance

The governor's `baseline_distance_per_frame_m` is the allowed travel per
processed vision frame. Change it only from measured real-vehicle control and
stopping performance.
