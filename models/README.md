# Model artifacts

Large model weights are intentionally not committed. The deployment manifest
expects `models/segformer-b0-ade20k.onnx` and its FP16 derivative
`models/segformer-b0-ade20k-fp16.onnx` for road segmentation, plus
`models/yolo11n.onnx` for the initial COCO detector. Native Mac acceleration
uses whole-model `.mlpackage` exports and smoke-tested `.mlmodelc` directories;
these large artifacts are also ignored by Git.

Install the optional runtimes:

```bash
python -m pip install -e '.[pretrained,onnx]'
```

Export SegFormer in its isolated process. This command must not be wrapped by a
script that imports `jetracer_sim` or `cv2` first:

```bash
.venv/bin/python tools/export_segformer_onnx.py
```

Create the FP16 derivative in a second ONNX-only process:

```bash
.venv/bin/python tools/convert_segformer_fp16.py
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

## Native Core ML

Install `coremltools` for the isolated conversion process:

```bash
.venv/bin/python -m pip install -e '.[coreml-export]'
```

Export both FP32 and FP16 MLPrograms. This process imports PyTorch and must not
import `jetracer_sim` or OpenCV:

```bash
.venv/bin/python tools/export_segformer_coreml.py
```

Compile and smoke-test the packages with Apple's CoreML framework:

```bash
.venv/bin/python tools/compile_coreml_models.py
```

The second command builds a small Objective-C++ helper and executes every model
in that separate process. A crash or failed prediction leaves no validation
record, so the simulator refuses to load the artifact. Successful validation
writes a SHA-256 fingerprint beside each compiled model.
