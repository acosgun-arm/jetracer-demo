# Jetson deployment and recovery

The deployment service is deliberately a **standby verifier**, not an
autonomous-driving service. On reboot it verifies the promoted release and a
current passing hardware preflight. Its systemd sandbox has
`PrivateDevices=true`, and its command line contains neither `--drive` nor
`--explicit-arm`. Driving is always a separate foreground action.

All commands below are run from the repository root on the Jetson. They do not
apply to the Mac simulator environment.

## 1. Build a target-native wheelhouse

Create the wheelhouse on the Jetson so every binary wheel matches its Python,
Linux, and aarch64 ABI. Start from the checked Jetson software baseline:

```bash
mkdir -p build/wheelhouse
python3 -m pip wheel . --wheel-dir build/wheelhouse
```

Add the selected NVIDIA/JetPack-compatible ONNX Runtime or other vendor wheels
to the same directory. Do not mix CPU and GPU builds of the same distribution:
release creation rejects duplicate distribution names. TensorRT supplied by
JetPack remains a system dependency and is captured by the software/preflight
reports; Python dependencies in the wheelhouse are fully hashed and installed
offline.

## 2. Create and prepare an immutable release

Choose a unique release ID. Creation copies only the configured inputs, writes
an absolute runtime preflight/bring-up binding into `deployed.json`, hashes all
source, model, config, and wheel files, then writes `requirements.lock`:

```bash
RELEASE_ID=2026-08-01.1
.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json create \
  --release-id "$RELEASE_ID" \
  --wheelhouse build/wheelhouse

.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json prepare \
  --release-id "$RELEASE_ID"

.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json verify \
  --release-id "$RELEASE_ID" --require-prepared
```

Preparation creates a release-local virtual environment with `--no-index` and
`--require-hashes`, runs `pip check`, and records the complete installed
distribution list. A failed preparation is retained for diagnosis; create a
new release ID after correcting the wheelhouse rather than mutating it.

## 3. Promote without arming

Promotion verifies the prepared release and atomically updates `current`. The
old current release becomes `previous`; no release directory is deleted.

```bash
.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json promote \
  --release-id "$RELEASE_ID"

.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json status
```

The promoted platform is also selectable directly through the existing single
platform switch:

```bash
export JETRACER_PLATFORM_CONFIG="$PWD/build/deployment/current/source/configs/platforms/deployed.json"
```

## 4. Install the standby-only reboot service

Rendering only writes a reviewable unit; it does not install, enable, or start
anything:

```bash
mkdir -p build/deployment
.venv/bin/python tools/render_systemd_unit.py \
  --config configs/deployment.json \
  --service-user jetracer \
  --output build/deployment/jetracer-standby.service
```

Inspect the file and confirm its `ExecStart` contains `--standby --watch`, its
device sandbox is enabled, and the paths/user match the Jetson. Only after a
passing hardware preflight exists may an administrator install it:

```bash
sudo install -m 0644 build/deployment/jetracer-standby.service \
  /etc/systemd/system/jetracer-standby.service
sudo systemctl daemon-reload
sudo systemctl enable --now jetracer-standby.service
```

If the release is invalid, unprepared, mismatched, or its preflight is missing,
failed, altered, or expired, standby exits nonzero and systemd retries. It never
opens a camera, GUI, or actuator. Its integrity-checked status is written to
`build/deployment/status.json`.

## 5. Explicit drive and safe stop

Do not use this command until the controller-specific driver exists and the
physical bring-up has reached an active moving stage. The released platform
must explicitly set `driver: "jetracer"`, `motors_enabled: true`, validated
state, and limits no higher than that stage. The runtime rechecks every gate
before it starts the common real-time application:

```bash
build/deployment/current/venv/bin/python \
  build/deployment/current/source/tools/run_deployed_runtime.py \
  --config configs/deployment.json --drive --explicit-arm
```

The drive runs in the foreground for the configured bounded duration. Ctrl-C
or SIGTERM follows the normal neutral/close path. From a second terminal, use
the PID- and command-verified stop operation:

```bash
build/deployment/current/venv/bin/python \
  build/deployment/current/source/tools/run_deployed_runtime.py \
  --config configs/deployment.json --safe-stop
```

Safe stop sends SIGTERM and waits for the configured deadline. It intentionally
does not use SIGKILL because bypassing vehicle cleanup is unsafe. Retain an
independent physical emergency stop for all moving tests.

Telemetry is preserved under `build/deployment/logs/` with the release ID and
UTC timestamp. Each release retains the exact configs, model artifacts,
wheelhouse, manifest, lock, and installed-distribution record used for it.
TensorRT engine caches are redirected outside the immutable release to
`build/deployment/cache/<release-id>/tensorrt/`.

Stop the standby verifier before a timed drive so its periodic integrity scan
cannot contend with inference, then restart it afterward:

```bash
sudo systemctl stop jetracer-standby.service
# run the explicit foreground drive command above
sudo systemctl start jetracer-standby.service
```

## 6. Rollback and recovery drill

Rollback verifies both releases, swaps `current` and `previous`, and does not
start the runtime:

```bash
.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json rollback
sudo systemctl restart jetracer-standby.service
.venv/bin/python tools/manage_deployment.py \
  --config configs/deployment.json status
```

Practice these failure cases before floor driving: corrupted release file,
expired preflight, absent camera, missing model provider, interrupted drive,
and rollback after a deliberately bad promotion. The expected result is either
verified standby or a fail-closed nonzero service—never automatic motor output.
