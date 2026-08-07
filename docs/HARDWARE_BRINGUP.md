# JetRacer hardware bring-up

Items 3 through 8 are implemented as fail-closed software gates. Their physical
acceptance remains pending until the delivered controller, cameras, state
sources and Jetson runtime can be measured. None of the commands in this guide
creates a GUI window. Inventory and validation commands do not command motors.

## 1. Camera inventory and calibration

List Jetson camera capabilities without opening a video stream:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/probe_jetson_cameras.py \
  --output build/hardware/camera-inventory.json inventory
```

After selecting a device, add `--device /dev/videoN` to query its exact V4L2
format and control tables. Capture cadence can then be measured headlessly. The
command opens the camera but creates no window:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/probe_jetson_cameras.py \
  --output build/hardware/elp-measurement.json measure \
  --profile elp_112 --device 0 --backend v4l2 \
  --calibration-report build/hardware/elp-calibration.json
```

The generated record contains these acceptance fields:

```json
{
  "duration_s": 10.0,
  "delivered_frames": 1200,
  "dropped_frames": 0,
  "width": 1920,
  "height": 1200,
  "pixel_format": "MEASURED_FORMAT",
  "capture_buffer_frames": 1,
  "calibration_rms_reprojection_error_px": 0.5
}
```

Validate that record against the selected profile:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/probe_jetson_cameras.py \
  --output build/hardware/elp-acceptance.json \
  validate-measurement --profile elp_112 \
  --measurement build/hardware/elp-measurement.json
```

The requested format, intrinsics, distortion, mounting transform and exposure
values in `configs/hardware/cameras.json` remain provisional until replaced by
these measurements.

Runtime transport and mode selection live in `configs/cameras/`. The ELP and
IMX219 files select AVFoundation, V4L2 or Jetson GStreamer independently of the
vehicle platform. To add another camera, add its physical measurement entry to
`configs/hardware/cameras.json`, create one runtime-profile JSON, and point the
platform's `camera.profile_config` and `camera.mode_id` at it.

Measure the camera optical centre from the rear-axle midpoint projected onto
the ground: `x_m` is forward, `y_m` is left, and `z_m` is upward. Record roll,
downward pitch, and yaw in radians. Set the mount `status` to `measured` only
when all six values have been recorded; the real runtime then overrides the
simulator's explicitly provisional mount transform.

## 2. Controller and state identification

The shipped actuator profile deliberately has no guessed controller endpoints:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/validate_actuator_profile.py \
  --output build/hardware/actuator-profile.json
```

This command must report blocked until the manufacturer, model, electrical
interface, steering endpoints, throttle endpoints and direction are measured.
The common actuator layer already enforces neutral-first, neutral-last,
configured limits, emergency stop and a command watchdog. A board-specific
transport is added only after the controller is identified.

If the kit exposes feedback, prefer it over the provisional command-response
speed estimate. Compare the selected state source with a reference measurement
using:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/validate_vehicle_state.py \
  --measurements build/hardware/speed-measurements.json \
  --output build/hardware/speed-validation.json
```

Each input measurement contains `timestamp_s`, `estimated_speed_mps`,
`reference_speed_mps` and `latency_s`. Motion remains blocked while
`validated_for_motion` is false or a state sample exceeds its configured age.

## 3. Jetson model deployment

The deployment gate verifies platform, artifact SHA-256, isolated runtime
providers and Jetson-local benchmarks. Unsupported CoreML variants are hidden:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/check_model_deployment.py \
  --output build/hardware/model-deployment.json
```

The current ONNX hashes are recorded, with separate CPU and TensorRT execution-
provider variants. All remain unselectable on a Jetson until sustained
benchmarks from that Jetson are added to the benchmark registry. The INT8 entry
is explicitly disabled until a representative calibration set produces a
validated artifact and hash. CPU remains the fallback; CUDA and TensorRT are
used only when actually reported by the runtime.

Generate the disabled INT8 candidate only after exporting a representative
dataset. Quantization is calibration, not model training, and runs in an
isolated process:

```bash
.venv-jetson/bin/python tools/quantize_segformer_int8.py \
  --dataset build/datasets/representative
```

The sidecar records source/target hashes, dataset-manifest hash, deterministic
sample selection and quantization settings. Add the resulting hash to the model
manifest and remove its disabled reason only after accuracy validation.

Benchmark segmentation variants with `tools/benchmark_model_variants.py` and
merge detector measurements into the same registry with:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/benchmark_detector_variants.py \
  --profile elp --model yolo11n-coco-tensorrt-fp16
```

## 4. Unified preflight

Run the single non-driving preflight after camera, controller, state and model
records have been completed:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  tools/hardware_preflight.py \
  --camera-measurement build/hardware/elp-measurement.json \
  --output build/hardware/preflight.json
```

It covers the Jetson software stack, camera acceptance, actuator identity and
calibration, dry-run output, state validation, model deployment, exact speed
certification, disk space, power mode and temperature. The report records that
no GUI, camera stream or physical output was opened. It is integrity-checked
and expires after the configured interval. A failed, stale, edited,
wrong-platform or superseded report cannot authorize motion.

## 5. Ordered physical stages

Initialize a persistent stage record:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python tools/bringup.py \
  --platform-id jetracer-pro init --state build/hardware/bringup.json
```

Begin the electronics-only stage, collect its evidence, then record its result:

```bash
PYTHONPATH=build-jetson/python .venv-jetson/bin/python tools/bringup.py \
  --platform-id jetracer-pro begin \
  --state build/hardware/bringup.json --stage electronics_only

PYTHONPATH=build-jetson/python .venv-jetson/bin/python tools/bringup.py \
  --platform-id jetracer-pro record \
  --state build/hardware/bringup.json --stage electronics_only \
  --outcome pass --evidence build/hardware/electronics-evidence.json
```

Moving stages also require `--preflight build/hardware/preflight.json` when
they are begun. Evidence must identify the stage, retain emergency-stop
availability, pass every named check, and prove that the configured stage speed
and steering limits were not exceeded. Stages cannot be skipped:

1. Electronics only
2. Wheels raised
3. Restrained low speed
4. Open-floor manual stop
5. Waveshare mat lane following
6. Stop signs
7. Obstacle avoidance

The platform profile must point to the current preflight and bring-up state
before the physical driver can be selected. It must also use limits no greater
than the active stage. The supplied profile stays on `dry_run` with motors
disabled until the missing hardware evidence is available.
