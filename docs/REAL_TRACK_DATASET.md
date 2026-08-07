# Real-track capture datasets

`configs/real_tracks.json` maps concrete track profile IDs to independent
dataset manifests. The initial profiles are `waveshare` and `floor`; both are
valid but report `awaiting_capture`. Large media and annotations stay under the
selected manifest's ignored `media/` and `annotations/` folders.

Select the live capture track with `capture.track_profile_id` in the platform
config. The browser capture panel displays the active concrete track. Offline
tools accept the same selection, for example:

```bash
.venv/bin/python tools/evaluate_real_track_dataset.py \
  --track-profile floor --overwrite
```

To add a track, create its manifest and add one entry to
`configs/real_tracks.json`; capture and evaluation code needs no modification.
Keep captures from different physical tracks in separate manifests even when
their markings look alike.

## Capture protocol

Use the final camera mount and retain native frames. The primary ELP mode is
1280×720 at 200 Hz; 1920×1200 at 120 Hz is the secondary detail mode. Record
fixed exposure and white balance when possible, and retain short auto-exposure
clips for transition testing.

Cover straight sections, both curves and full laps across diffuse daylight,
strong shadows, warm and cool indoor lighting, low light and glare. Capture
empty-road scenes and the cylinder at the centre, left and right of the road.
Include lateral and heading errors rather than recording only perfect driving.

Do not tune against the benchmark split. Calibration images determine camera
and colour parameters, development images support implementation, and the
benchmark split remains held out.

Before normal capture, photograph a ruler and colour/checkerboard target and
record the measured road width, camera height, pitch, rotation and lateral
offset in the manifest.

## Browser capture UI

Connect the ELP camera and run:

```bash
.venv/bin/python examples/realtime_demo.py \
  --platform-config configs/platforms/mac-elp.json
```

Open `http://127.0.0.1:8765` manually. The dashboard can show the raw camera
image or the processed overlays. Its capture panel writes raw PNG snapshots and
MP4 videos to `datasets/real_track/media/`, fingerprints them, and adds the
selected capture metadata to this manifest. Recording is asynchronous; watch
the displayed dropped-frame count when validating whether the configured codec
can sustain the requested 200 Hz mode.

The server is localhost-only and creates no native GUI window. Grant macOS
camera permission while unlocked on the first run; do not use `--open-browser`
if the screen may be locked.

## Register media

Place a lossless image or video inside `datasets/real_track/media/`. Semantic
masks are single-channel PNG files with these class IDs:

- `0`: background
- `1`: drivable surface
- `2`: orange boundary
- `3`: magenta cylinder

Registering a file records its SHA-256 identity. Labelled video frames should
be extracted losslessly and registered as image captures.

```bash
.venv/bin/python tools/register_real_track_capture.py \
  datasets/real_track/media/daylight-straight-empty-001.png \
  --capture-id daylight-straight-empty-001 \
  --track-profile waveshare \
  --split calibration --media-type image \
  --camera-mode elp_720p_200 --lighting daylight_diffuse \
  --track-section straight --scene-type empty_road \
  --semantic-mask \
  datasets/real_track/annotations/daylight-straight-empty-001.png
```

## Validate and evaluate

Validation is headless and checks coverage, dimensions, video FPS, masks,
paths and content hashes:

```bash
.venv/bin/python tools/evaluate_real_track_dataset.py --overwrite
```

Generate train-free HSV profiles from the calibration split and score them on
development and held-out benchmark images:

```bash
.venv/bin/python tools/calibrate_real_track_colours.py --overwrite
```

Prepare labelled benchmark stills for any configured segmentation model:

```bash
.venv/bin/python tools/prepare_real_track_segmentation.py --overwrite
.venv/bin/python tools/evaluate_segmentation.py \
  --dataset build/datasets/real-track-evaluation
```

Replay each registered video through the high-rate inference benchmark:

```bash
.venv/bin/python tools/benchmark_recorded_clip.py \
  datasets/real_track/media/CLIP.mov
```
