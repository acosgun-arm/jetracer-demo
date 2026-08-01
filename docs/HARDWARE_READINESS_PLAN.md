# JetRacer hardware-readiness plan

The work is deliberately ordered so that unsafe hardware output cannot be
enabled before its dependencies are measured and tested. Each item has a
concrete completion gate.

## 1. Runtime safety and shutdown

Implement a command watchdog in the common actuator layer, neutral output on
start/timeout/error/exit, latched emergency-stop state, and SIGINT/SIGTERM
handling in the real-time process. Keep every timeout in the selected platform
or runtime configuration.

Acceptance: an interrupted or stalled control loop reaches neutral without
waiting for inference or UI cleanup; commands are rejected after watchdog
expiry; automated tests exercise the behavior with no physical I/O.

Status: complete. Both platform profiles configure a 0.25-second watchdog.
Neutral commands disarm it; a missed non-neutral command deadline latches an
emergency stop. The real-time process converts SIGINT/SIGTERM into an orderly
platform stop, and headless tests cover watchdog expiry and an actual SIGTERM.
This software watchdog does not replace a controller-side hardware watchdog.

## 2. Jetson software baseline

Record the delivered Jetson module, JetPack/L4T, CUDA, TensorRT, Python and
OpenCV/GStreamer versions. Add a supported-version manifest, reproducible
bootstrap script, dependency preflight and CPU-only fallback. Audit the current
Python requirement and native build flags against the delivered JetPack image.

Acceptance: a clean Jetson image can build/import the package and run the
headless simulator and model preflight from documented commands.

Status: software implementation complete; hardware acceptance pending. The
versioned provisional manifest, dry-run-first bootstrap and stdlib-only JSON
preflight are in place. Python 3.10 and CMake 3.22 are now supported, and a
headless build no longer requires OpenCV HighGUI. Exact module, JetPack/L4T,
CUDA, TensorRT, OpenCV and GStreamer values remain intentionally unset until
they can be read from the delivered board.

## 3. Camera integration and calibration

Probe both the ELP UVC and IMX219 cameras on the Jetson without opening a GUI.
Record exact V4L2/GStreamer modes, pixel formats, rational frame rates, buffer
behavior and sustained frame-delivery statistics. Add per-camera profiles,
intrinsic/distortion calibration, mounting transforms and exposure controls.

Acceptance: each selected profile sustains its required rate for a bounded test,
reports dropped/replaced frames, and passes a calibration reprojection-error
gate. Unverified modes remain marked provisional.

Status: software implementation complete; hardware acceptance pending. ELP and
IMX219 records, headless V4L2/GStreamer inventory and bounded capture,
cadence/drop thresholds, buffer limits and calibration reprojection gates are implemented. Jetson modes,
intrinsics, distortion, mounts and exposure remain explicitly unmeasured.

## 4. Steering and motor actuator driver

Identify the supplied motor/servo controller and electrical interface before
adding its backend. Implement neutral-first initialization, direction and
channel mapping, calibrated steering/throttle conversion, configured limits,
the common watchdog, emergency stop and neutral-on-close. Keep motor enablement
as an explicit config interlock.

Acceptance: wheels-up tests prove correct direction, neutral, endpoints,
watchdog timeout and process-exit behavior before floor testing is allowed.

Status: safety architecture complete; controller-specific implementation and
physical acceptance pending. Calibrated mapping uses an injected transport and
enforces neutral-first/last, limits, watchdog and emergency stop. The supplied
controller profile is unidentified, uncalibrated and cannot enable output.

## 5. Vehicle state and speed estimation

Determine whether the delivered kit exposes wheel encoders or other feedback.
Prefer measured speed; otherwise build and validate an estimated state source
from throttle calibration and optional camera/IMU evidence. Report timestamp,
freshness, source and quality with every sample.

Acceptance: speed error and latency meet a recorded low-speed test gate. The
physical actuator remains blocked while state is unavailable or stale.

Status: software implementation complete; sensor inventory and physical
validation pending. State samples expose source, quality, confidence, sequence
and age. A configurable command-response estimate is available at low
confidence, while saved reference-speed tests gate error and latency. Physical
motion requires a fresh source explicitly marked validated.

## 6. Jetson inference backends

Deploy the off-the-shelf segmentation and YOLO artifacts through ONNX Runtime
and TensorRT where supported. Generate FP32/FP16/INT8 variants reproducibly,
record artifact hashes and calibration provenance, benchmark warm and sustained
latency, and retain live model switching.

Acceptance: every selectable model has a validated artifact and benchmark for
the delivered Jetson; unsupported variants are hidden rather than failing at
drive time.

Status: deployment gate complete; Jetson runtime validation and benchmarks
pending. ONNX artifacts have recorded SHA-256 values, runtime providers are
probed in an isolated process, CoreML is excluded from Jetson, and the real
model menu admits only target-compatible, hashed and Jetson-benchmarked models.
FP32/FP16 CPU and TensorRT-provider candidates are configured; deterministic
INT8 static quantization records calibration-dataset provenance and stays
disabled until its artifact and accuracy are validated.

## 7. Hardware preflight and observability

Create one non-driving preflight covering configuration, model artifacts,
camera negotiation, actuator dry-run, state freshness, storage, power mode,
temperature and disk space. Extend telemetry with platform identity, capture
health, watchdog state, command/state age, inference latency and thermal data.

Acceptance: driving cannot be armed after a failed mandatory check, and the
result is saved as a machine-readable report.

Status: complete in software. The unified non-driving report covers every
listed category, records safety side effects, is integrity-checked and expires.
The current report correctly fails because physical evidence is absent.
Telemetry includes platform/capture/watchdog health, command and state age,
inference latency, confidence and cached thermal readings.

## 8. Staged vehicle bring-up

Run an explicit sequence: electronics-only, wheels raised, restrained low
speed, open-floor manual stop, lane following on the Waveshare mat, stop signs,
then obstacle avoidance. Start with conservative configured speed/steering
limits and increase them only from recorded evidence.

Acceptance: each stage has a saved result and must pass before the next stage is
enabled. Emergency stop remains available throughout.

Status: workflow implementation complete; physical execution pending. The
integrity-checked state machine prevents skipped stages, requires a current
preflight for moving stages, validates evidence and emergency-stop availability,
and exposes per-stage speed and steering ceilings to the runtime arm gate.

## 9. Deployment and recovery

Package a pinned runtime for the Jetson, select the platform with one config or
environment variable, add supervised startup only after preflight, and preserve
logs/config/artifact versions. Define rollback and safe manual shutdown.

Acceptance: reboot-to-ready and rollback are repeatable, but motors never arm
automatically without successful preflight and an explicit enable action.

Status: complete in software; Jetson recovery drill pending. Releases contain
hashed source/config/model artifacts, a hash-locked offline wheelhouse,
integrity-checked preparation metadata, and atomic current/previous promotion.
The rendered reboot service is a device-sandboxed standby verifier only.
Driving remains a separate explicit-arm foreground command gated by the
physical driver interlock, preflight, and active bring-up stage. Safe stop uses
bounded SIGTERM without an unsafe forced-kill fallback. Deployment, rollback,
log retention, and recovery drills are documented in `DEPLOYMENT_RECOVERY.md`.
