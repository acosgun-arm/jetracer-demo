# Model artifacts

Large model weights are intentionally not committed. The active platform
expects Cityscapes SegFormer-B0/B1 ONNX road models plus `models/yolo11n.onnx`
for COCO detection. Native Mac acceleration uses whole-model `.mlpackage`
exports and smoke-tested `.mlmodelc` directories; these artifacts are ignored
by Git.

Install the optional runtimes:

```bash
python -m pip install -e '.[pretrained,onnx]'
```

Export both pinned Cityscapes candidates in isolated processes. These commands
must not be wrapped by a script that imports `jetracer_sim` or `cv2` first:

```bash
.venv/bin/python tools/export_segformer_onnx.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b0-cityscapes-cpu-fp32
.venv/bin/python tools/export_segformer_onnx.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b1-cityscapes-cpu-fp32
```

Run a lane-only closed-loop comparison without detector overhead:

```bash
.venv/bin/python tools/run_driving_benchmarks.py \
  --platform configs/platforms/sim.json --perception actual \
  --model-key 2 --no-detector --scenario lane \
  --track waveshare_3x2 --laps 1 --maximum-time 90
```

Export the upstream YOLO11n COCO checkpoint with a fixed 640-pixel input:

```bash
python -m pip install ultralytics
yolo export model=yolo11n.pt format=onnx imgsz=640 dynamic=False simplify=False
mv yolo11n.onnx models/yolo11n.onnx
```

Then run the Jetson-targeted headless readiness check from the repository root:

```bash
PYTHONPATH=build-python/python python tools/check_model_deployment.py
```

It deliberately remains blocked on macOS and until the target Jetson reports
its execution providers and has target-local segmentation and detector
benchmarks. CoreML variants are hidden on Jetson; CPU and TensorRT execution-
provider variants are evaluated independently.

The SegFormer exporter downloads through the Hugging Face cache on first use,
validates the `road` class mapping, checks the ONNX graph, and writes a checksum
sidecar next to the ignored model artifact. The converter preserves FP32 model
inputs and outputs, validates the converted graph, and writes its own checksum
sidecar. Neither command should share a process with the simulator or OpenCV on
macOS.

PIDNet-S can be reproduced for Jetson experiments with:

```bash
.venv/bin/python tools/export_pidnet_onnx.py --opset 14 --overwrite
```

It reaches 28.03 FPS with TensorRT FP16 on the Nano, but is not selectable: its
Waveshare synthetic-clip road IoU is only 0.071. The retained evidence is in
`benchmarks/jetson_pidnet_tensorrt.json`.

## Colour-lane backend

Model key `10` thresholds the simulated Waveshare white boundaries in HSV,
rejects outliers with iterative polynomial fitting, and emits the same road
mask interface as the neural models. Its shared Python/C++ profile, including
optional normalized bird's-eye points, is
`configs/color_lane/waveshare-sim-white.json`.
The current IPM is disabled until a real camera-to-ground homography is
measured. Benchmark evidence is in `benchmarks/color_lane_waveshare.json`.

## Native Core ML

Core ML 9 does not provide its required native conversion extensions for
Python 3.14. Create a disposable Python 3.13 exporter environment under the
ignored build directory:

```bash
brew install python@3.13
python3.13 -m venv build/coreml-export-venv
build/coreml-export-venv/bin/python -m pip install \
  -r requirements/coreml-export.txt
```

Export one pinned FP16 MLProgram at a time. This process imports PyTorch and
must not import `jetracer_sim` or OpenCV:

```bash
build/coreml-export-venv/bin/python tools/export_segformer_coreml.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b0-cityscapes-coreml-fp16-256
build/coreml-export-venv/bin/python tools/export_segformer_coreml.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b0-cityscapes-coreml-fp16-384
build/coreml-export-venv/bin/python tools/export_segformer_coreml.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b0-cityscapes-coreml-fp16
build/coreml-export-venv/bin/python tools/export_segformer_coreml.py \
  --models configs/road_segmentation_models.json \
  --model-id segformer-b1-cityscapes-coreml-fp16
```

Compile and smoke-test the packages with Apple's CoreML framework:

```bash
.venv/bin/python tools/compile_coreml_models.py \
  --models configs/road_segmentation_models.json
```

The second command builds a small Objective-C++ helper and executes every model
in that separate process. A crash or failed prediction leaves no validation
record, so the simulator refuses to load the artifact. Successful validation
writes a SHA-256 fingerprint beside each compiled model.

The simulator prefers the balanced 384-pixel B0 variant. Select key `3` for
the approximately 200 FPS 256-pixel demo or key `9` for the 512-pixel variant.
