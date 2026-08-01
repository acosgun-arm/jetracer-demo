# Jetson software setup

The current software baseline is provisional because the exact Jetson module
and factory JetPack image have not arrived. It does not guess or replace the
NVIDIA stack. The delivered versions will be captured in a preflight report and
then promoted into `configs/jetson_software_baseline.json`.

All commands below are headless. They do not open a camera or GUI and cannot
energise the vehicle.

## Before the hardware arrives

Validate the manifest and inspect this development host:

```bash
python3 tools/check_jetson_compatibility.py \
  --output /tmp/jetracer-development-compatibility.json
python3 tools/bootstrap_jetson.py
```

The first command reports `development_host` on a compatible non-Jetson
machine. The second prints the future Jetson changes but performs a dry run.

## On the delivered Jetson

First save an unmodified inventory. A blocked result is expected before build
dependencies are installed:

```bash
python3 tools/check_jetson_compatibility.py --strict-target \
  --output /tmp/jetracer-jetson-before-bootstrap.json
```

Review the bootstrap plan, then explicitly apply it:

```bash
python3 tools/bootstrap_jetson.py
python3 tools/bootstrap_jetson.py --apply
```

The bootstrap installs only host build dependencies, creates
`.venv-jetson`, and installs this project with the native GUI CLI disabled. It
does not install or modify JetPack, CUDA, TensorRT, camera drivers, or motor
drivers.

Run the strict report again and verify the package and headless simulator:

```bash
.venv-jetson/bin/python tools/check_jetson_compatibility.py --strict-target \
  --output /tmp/jetracer-jetson-after-bootstrap.json
.venv-jetson/bin/python -c "import jetracer_sim"

cmake -S . -B build-jetson -G Ninja \
  -DJETRACER_SIM_BUILD_NATIVE_CLI=OFF \
  -DJETRACER_SIM_BUILD_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-jetson
ctest --test-dir build-jetson --output-on-failure

PYTHONPATH=build-jetson/python .venv-jetson/bin/python \
  examples/realtime_demo.py \
  --platform-config configs/platforms/sim.json \
  --model-config configs/demo_models.json \
  --benchmark-registry benchmarks/demo_model_benchmarks.json \
  --headless --duration 1 --no-log
```

Before declaring this baseline supported, copy the measured module,
JetPack/L4T, CUDA, TensorRT, OpenCV and GStreamer versions from the report into
the manifest, then retain both before/after reports with the hardware bring-up
records.
