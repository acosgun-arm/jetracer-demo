# Jetson software setup

The validated Jetson Nano baseline is JetPack 4.6.6 / L4T 32.7.6, CUDA 10.2,
TensorRT 8.2.1, GCC 7.5, CMake 3.10, and OpenCV 4.1. The native deployment path
is C++17 and avoids coupling the future inference loop to the Nano's system
Python.

All commands below are headless. They do not open a camera or GUI and cannot
energise the vehicle.

## Deploy from the Mac

The deployment helper synchronizes source without model artifacts or build
outputs, builds natively, runs the simulator core test, and runs a CUDA/TensorRT
self-test. It never opens the camera or actuator interfaces:

```bash
tools/deploy_jetson_native.sh
```

Override the target when its address changes:

```bash
JETRACER_JETSON_HOST=jetson@192.168.50.195 \
  tools/deploy_jetson_native.sh
```

The successful probe prints `"ready": true`, the detected TensorRT/CUDA
versions, and `"actuators_accessed": false`.

With a UVC camera connected, enumerate its native formats and frame rates
without displaying or recording frames:

```bash
ssh jetson@192.168.50.195 \
  /home/jetson/jetracer-demo/build-jetson-native/jetracer-jetson-runtime \
  --camera-probe /dev/video0
```

Run the reproducible 200 Hz transport benchmark (compressed frames are
discarded without decoding or recording):

```bash
ssh jetson@192.168.50.195 \
  /home/jetson/jetracer-demo/build-jetson-native/jetracer-jetson-runtime \
  --camera-benchmark /dev/video0 1280 720 200 MJPG 10 4 5000 1000
```

The measured hardware result is retained in
`benchmarks/jetson_elp_camera_transport.json`.

Benchmark NVIDIA JPEG decode, the configured 180-degree correction, and resize
to the 512x512 TensorRT input while retaining the result in NVMM GPU memory:

```bash
ssh jetson@192.168.50.195 \
  /home/jetson/jetracer-demo/build-jetson-native/jetracer-jetson-runtime \
  --preprocess-benchmark /dev/video0 1280 720 200 512 512 2 10 5000 1000
```

## Current scope

This establishes native compilation, NVIDIA runtime probing, and lossless UVC
transport at 200 Hz. MJPEG decoding, TensorRT engine loading,
preprocessing/postprocessing, and the fail-closed actuator adapter are the next
deployment layers; motors remain disabled until those layers pass hardware
bring-up.

## TensorRT compatibility

TensorRT 8.2 cannot parse the opset-17 SegFormer `LayerNormalization`
operator. Export the Jetson variant with opset 14, which decomposes it into
supported primitives:

```bash
.venv/bin/python tools/export_segformer_onnx.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b0-cityscapes-cpu-fp32 \
  --output models/segformer-b0-cityscapes-512-opset14.onnx \
  --opset 14 --overwrite
```

The engine and inference measurements are recorded in
`benchmarks/jetson_segformer_tensorrt.json`. At MAXN the 512x512 FP16 model
delivers 3.91 FPS, so it is retained as a high-quality slow option rather than
the high-speed default.

PIDNet-S was also exported and benchmarked as a TensorRT-friendly CNN. It
reaches 28.03 FPS, but fails the current synthetic Waveshare quality gate due
to 7.1% road recall and therefore is not exposed as a selectable model. See
`benchmarks/jetson_pidnet_tensorrt.json` for the reproducible result.

## Native color-lane benchmark

The classical Waveshare backend shares one profile between Python and C++.
Benchmark its complete ELP path without accessing actuators:

```bash
build-jetson-native/jetracer-jetson-runtime \
  --color-lane-camera-benchmark \
  configs/color_lane/waveshare-sim-white.json \
  /dev/video0 1280 720 200 2 10 5000 1000
```

On the Nano this delivered 199.28 FPS with no camera-buffer gaps. Native lane
processing averaged 3.78 ms and had 264.72 FPS standalone capacity. This white
profile is for simulation validation; a real orange-line profile must be
calibrated from physical-track images before driving.

## Motor-disabled controller replay

Replay the deterministic ELP/Waveshare clip through native colour fitting,
ground projection, adaptive pure pursuit, and the speed governor:

```bash
build/native-shadow/jetracer-shadow-replay \
  configs/color_lane/waveshare-sim-white.json \
  configs/native_shadow_controller.json \
  benchmarks/synthetic_clips/waveshare_3x2-elp-20260801-214642/rgb.mp4 \
  build/shadow-replay/waveshare-elp.jsonl
```

The shadow configuration requires `"actuator_mode": "disabled"`; the binary
contains only a telemetry sink and no PWM/I²C actuator adapter. Its camera
calibration is synthetic-only and must not be used to enable real motion.
