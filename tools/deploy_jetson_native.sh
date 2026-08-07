#!/bin/sh
set -eu

JETRACER_JETSON_HOST=${JETRACER_JETSON_HOST:-jetson@192.168.50.195}
JETRACER_JETSON_DIR=${JETRACER_JETSON_DIR:-/home/jetson/jetracer-demo}

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

ssh "$JETRACER_JETSON_HOST" "mkdir -p '$JETRACER_JETSON_DIR'"
rsync -az \
  --exclude='.git/' \
  --exclude='.venv*/' \
  --exclude='build*/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='datasets/' \
  --exclude='benchmark_results/' \
  --exclude='artifacts/' \
  --exclude='*.onnx' \
  --exclude='*.engine' \
  --exclude='*.mlpackage/' \
  "$repo_dir/" "$JETRACER_JETSON_HOST:$JETRACER_JETSON_DIR/"

ssh "$JETRACER_JETSON_HOST" "
  set -eu
  cd '$JETRACER_JETSON_DIR'
  mkdir -p build-jetson-native
  cd build-jetson-native
  cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DJETRACER_SIM_BUILD_PYTHON=OFF \
    -DJETRACER_SIM_BUILD_NATIVE_CLI=OFF \
    -DJETRACER_SIM_BUILD_TESTS=ON \
    -DJETRACER_SIM_BUILD_JETSON_RUNTIME=ON
  cmake --build . -- -j2
  ctest --output-on-failure
  ./jetracer-jetson-runtime --self-test
"
