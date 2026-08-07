# Race optimization log — 2026-08-03

## Objective and fixed protocol

- Window: 01:30–09:00 Europe/London.
- Tracks: `waveshare_3x2`, `technical_chicane`, `tight_hairpin`.
- Camera source: simulated 200 Hz camera.
- Perception: best available validated model, SegFormer-B0 Core ML FP16 at
  384 px (model key 4).
- Scenario: lane following only; no detector, pedestrians, or stop signs.
- Requested cruise speed: 2.5 m/s.
- Default comparison stack: pure pursuit, temporal path filtering off,
  minimum-time racing line, curvature speed planner off.
- Development gate: three laps per track, zero off-road events on every track,
  and higher average speed on every track than the incumbent.
- Accepted candidates receive a longer confirmation run before replacing the
  incumbent.

## Experiments

Results are appended chronologically. Rejected experiments remain here so
later hypotheses can use their evidence.

### 00 — Cooled current-code baseline

Hypothesis: establish the incumbent under a fresh, thermally stable run before
changing control behaviour.

| Track | Mean speed (m/s) | Off-road | Mean deviation (m) | Segmentation FPS |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 1.0983 | 0 | 0.0424 | 129.5 |
| Technical chicane | 1.1095 | 0 | 0.0298 | 127.9 |
| Tight hairpin | 1.0780 | 0 | 0.0370 | 127.8 |

Macro mean speed: **1.0953 m/s**. Status: **incumbent**.

Reports: `build/benchmarks/race-opt-00-baseline-*.json`.

### 01 — Governor capacity factor 0.90 → 0.94

Hypothesis: the incumbent is frame-rate limited at approximately 1.15 m/s;
using 94% rather than 90% of measured inference capacity may raise speed while
remaining within the 0.010 m/frame perception-distance bound.

Result: **inconclusive and reverted**. The first candidate leg ran after the
extended curvature campaign and Core ML completion rate fell from 129.5 to
99.4 FPS. Waveshare speed consequently fell to 0.879 m/s with zero off-road
events. Remaining legs were stopped because thermal throttling, not the
governor parameter, dominated the comparison.

Evidence: `build/benchmarks/race-opt-01-governor-094-waveshare_3x2.json`.

Learning: actual-model throughput is an uncontrolled variable during a long
continuous campaign. Use deterministic benchmarked-latency perception for
development comparisons and reserve actual Core ML for cooled confirmation.

### 02 — Fixed-governor benchmark protocol

Material tooling change: added `--fixed-governor-fps` and
`--fixed-governor-latency-s` to `tools/run_control_benchmarks.py`. The real
Core ML model still supplies every road mask, while governor telemetry is held
at the cooled baseline (128 FPS, 12 ms). This separates controller changes
from thermal variation in commanded speed. The report records the override.

Validation baseline: Waveshare 1.0877, technical 1.1180, hairpin 1.0870 m/s;
zero off-road events; macro 1.0976 m/s. Actual completion FPS varied from
104.5 to 125.3 without materially moving commanded speed.

Reports: `build/benchmarks/race-opt-fixed-00-baseline-*.json`.

### 03 — Governor capacity factor 0.90 → 0.94, controlled retry

Hypothesis: 94% utilization raises the frame-rate speed ceiling from 1.152 to
1.203 m/s at 128 FPS while preserving the configured 0.010 m/frame bound.

| Track | Incumbent (m/s) | Candidate (m/s) | Change | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 1.0877 | 1.1309 | +4.0% | 0 |
| Technical chicane | 1.1180 | 1.1654 | +4.2% | 0 |
| Tight hairpin | 1.0870 | 1.1298 | +3.9% | 0 |

Development macro: **1.1420 m/s** (+4.0%). A reverse-order ten-lap
confirmation also had zero off-road events on every track: Waveshare 1.1804,
technical 1.1915, hairpin 1.1800 m/s; macro **1.1840 m/s**. The confirmation
remained safe with actual completion FPS as low as 96.6, exercising cached-path
propagation under load.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-01-governor-094-*.json` and
`build/benchmarks/race-opt-fixed-01-confirm10-governor-094-*.json`.

### 04 — Governor capacity factor 0.94 → 0.98

Hypothesis: use more of the configured 0.010 m/frame perception-distance
budget while retaining a 2% reserve.

| Track | 0.94 (m/s) | 0.98 (m/s) | Change | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 1.1309 | 1.173 | +3.7% | 0 |
| Technical chicane | 1.1654 | 1.212 | +4.0% | 0 |
| Tight hairpin | 1.1298 | 1.172 | +3.7% | 0 |

Development macro: approximately **1.186 m/s**, +3.8% over the previous
incumbent and +8.1% over the controlled baseline.

Ten-lap result: **rejected**. The tight hairpin recorded one off-road event at
1.203 m/s. Remaining confirmation legs were stopped after the strict gate had
failed. The 0.94 setting remains the incumbent.

Reports: `build/benchmarks/race-opt-fixed-02-governor-098-*.json`.
Confirmation evidence:
`build/benchmarks/race-opt-fixed-02-confirm10-governor-098-tight_hairpin.json`.

### 05 — Governor capacity factor 0.98 → 1.00

Hypothesis: remove the final 2% capacity reserve.

Result: **rejected**. Waveshare reached 1.193 m/s and technical reached
1.235 m/s without recovery, but the tight hairpin recorded one off-road event
at 1.116 m/s. The hairpin is the limiting track, and the configured
0.010 m/frame distance budget requires margin with the current controller.

Reports: `build/benchmarks/race-opt-fixed-03-governor-100-*.json`.

### 06 — Shorter speed-dependent lookahead at governor 0.98

Hypothesis: the hairpin failure at 0.98 is geometric understeer. At 1.2 m/s,
the original 0.24 s term produces approximately 0.59 m lookahead, too long for
the tight radius. Reduce only `speed_lookahead_s` from 0.24 to 0.18.

Ten-lap results:

| Track | 0.94 incumbent (m/s) | Candidate (m/s) | Mean deviation | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 1.1804 | 1.229 | 0.0321 m | 0 |
| Technical chicane | 1.1915 | 1.241 | 0.0199 m | 0 |
| Tight hairpin | 1.1800 | 1.229 | 0.0290 m | 0 |

Macro mean: approximately **1.233 m/s**, +4.1% over the confirmed 0.94
incumbent. The hairpin deviation fell from 0.0458 m in the failed 0.98 trial
to 0.0290 m. Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-04-lookahead-018-governor-098-*.json`.

### 07 — Full governor utilization after lookahead correction

Hypothesis: the 1.00 utilization failure was caused by hairpin understeer, not
the utilization value itself. Retest 1.00 with the accepted 0.18 s lookahead.

Ten-lap results: Waveshare **1.253**, technical **1.266**, hairpin **1.253
m/s**; zero off-road events on all tracks; macro approximately **1.257 m/s**
(+2.0% over experiment 06). Actual Core ML completion rate reached a low of
88.6 FPS on the hairpin without loss of control.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-05-lookahead-018-governor-100-*.json`.

### 08 — Distance per processed frame 0.0100 → 0.0105 m

Hypothesis: the corrected lookahead provides enough lateral-control margin to
raise the governor's frame-distance budget by 5%.

Ten-lap results: Waveshare **1.313**, technical **1.328**, hairpin **1.313
m/s**; zero off-road events; macro approximately **1.318 m/s** (+4.9% over
experiment 07 and about +20% over the controlled baseline).

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-06-distance-0105-*.json`.

### 09 — Distance per processed frame 0.0105 → 0.0110 m

Hypothesis: take the next 0.0005 m/frame step with the accepted shorter
lookahead.

Ten-lap results: Waveshare **1.373**, technical **1.390**, hairpin **1.373
m/s**; zero off-road events; macro approximately **1.379 m/s** (+4.6% over
experiment 08 and about +26% over the controlled baseline).

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-07-distance-0110-*.json`.

### 10 — Distance per processed frame 0.0110 → 0.0115 m

Ten-lap results: Waveshare **1.432**, technical **1.451**, hairpin **1.432
m/s**; zero off-road events; macro approximately **1.438 m/s** (+4.3% over
experiment 09 and about +31% over the controlled baseline).

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-08-distance-0115-*.json`.

### 11 — Distance per processed frame 0.0115 → 0.0120 m

Ten-lap results: Waveshare **1.491**, technical **1.513**, hairpin **1.491
m/s**; zero off-road events; macro approximately **1.498 m/s** (+4.2% over
experiment 10 and about +36% over the controlled baseline). Deviation is
increasing gradually, so subsequent increments retain the hairpin-first gate.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-09-distance-0120-*.json`.

### 12 — Distance per processed frame 0.0120 → 0.0125 m

Ten-lap results: Waveshare **1.549**, technical **1.574**, hairpin **1.549
m/s**; zero off-road events; macro approximately **1.557 m/s** (+3.9% over
experiment 11 and about +42% over the controlled baseline).

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-10-distance-0125-*.json`.

### 13 — Distance per processed frame 0.0125 → 0.0130 m

Result: **rejected**. The hairpin recorded one off-road event in 10 laps at
1.554 m/s. At this speed, the 0.18 s lookahead term again produces roughly
0.59 m total lookahead, matching the understeer region observed before
experiment 06. Other tracks were not run after the strict gate failed.

Evidence: `build/benchmarks/race-opt-fixed-11-distance-0130-tight_hairpin.json`.

### 14 — Lookahead 0.18 → 0.15 s at 0.0130 m/frame

Hypothesis: remove the renewed high-speed hairpin understeer by shortening the
speed-dependent lookahead.

Ten-lap results: Waveshare **1.608**, technical **1.635**, hairpin **1.608
m/s**; zero off-road events; macro approximately **1.617 m/s** (+3.9% over
experiment 12 and about +47% over the controlled baseline).

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-12-lookahead-015-distance-0130-*.json`.

### 15 — Distance per processed frame 0.0130 → 0.0135 m

Ten-lap results: Waveshare **1.666**, technical **1.696**, hairpin **1.666
m/s**; zero off-road events; macro approximately **1.676 m/s** (+3.7% over
experiment 14 and about +53% over the controlled baseline). Waveshare remained
safe with actual Core ML completion at 88.7 FPS.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-13-distance-0135-*.json`.

### 16 — Distance per processed frame 0.0135 → 0.0140 m

Ten-lap results: Waveshare **1.723**, technical **1.756**, hairpin **1.723
m/s**; zero off-road events; macro approximately **1.734 m/s** (+3.5% over
experiment 15 and about +58% over the controlled baseline). Mean center
deviation was 0.0409, 0.0283, and 0.0350 m respectively.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-14-distance-0140-*.json`.

### 17 — Invalid packaged-config fallback (excluded)

Two resumed commands accidentally omitted the explicit `--config` argument,
so they loaded the installed package's stale 0.24 s lookahead instead of the
working tree's incumbent 0.15 s value (or the intended 0.12 s candidate).
Both produced 16 hairpin exits at the 0.0145 m/frame setting. These results are
excluded from optimization decisions because two variables changed at once.

Evidence retained for audit:
`build/benchmarks/race-opt-fixed-15-distance-0145-tight_hairpin.json` and
`build/benchmarks/race-opt-fixed-16-lookahead-012-distance-0145-tight_hairpin.json`.

Learning: every campaign command must explicitly pin the driving, model, and
runtime config paths; report `configuration_path` is checked before scoring.

### 18 — Distance per processed frame 0.0140 → 0.0145 m, corrected run

Result: **rejected as-is**. With all config paths pinned and the incumbent
0.15 s lookahead, the hairpin recorded one off-road event in 10 laps. Mean
speed was 1.709 m/s after recovery, peak speed 1.856 m/s, and actual Core ML
completion 129.8 FPS. The strict zero-exit gate stopped the other tracks.

Evidence: `build/benchmarks/race-opt-fixed-17-distance-0145-tight_hairpin.json`.

### 19 — Lookahead 0.15 → 0.12 s at 0.0145 m/frame

Hypothesis: the 0.0145 m/frame failure is speed-dependent understeer, so a
shorter pure-pursuit lookahead should turn earlier and restore margin.

Ten-lap results: Waveshare **1.782**, technical **1.817**, hairpin **1.780
m/s**; zero off-road events; macro approximately **1.793 m/s** (+3.4% over
experiment 16 and about +63% over the controlled baseline). Mean center
deviation improved on all three tracks to 0.0254, 0.0183, and 0.0326 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-18-lookahead-012-distance-0145-*.json`.

### 20 — Distance per processed frame 0.0145 → 0.0150 m

Ten-lap results: Waveshare **1.838**, technical **1.877**, hairpin **1.836
m/s**; zero off-road events; macro approximately **1.850 m/s** (+3.2% over
experiment 19 and about +69% over the controlled baseline). Mean center
deviation was 0.0254, 0.0177, and 0.0324 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-19-distance-0150-*.json`.

### 21 — Distance per processed frame 0.0150 → 0.0155 m

Ten-lap results: Waveshare **1.894**, technical **1.937**, hairpin **1.893
m/s**; zero off-road events; macro approximately **1.908 m/s** (+3.1% over
experiment 20 and about +74% over the controlled baseline). Mean center
deviation was 0.0272, 0.0197, and 0.0319 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-20-distance-0155-*.json`.

### 22 — Distance per processed frame 0.0155 → 0.0160 m

Ten-lap results: Waveshare **1.950**, technical **1.996**, hairpin **1.948
m/s**; zero off-road events; macro approximately **1.965 m/s** (+3.0% over
experiment 21 and about +79% over the controlled baseline). Mean center
deviation rose to 0.0287, 0.0220, and 0.0362 m, signalling diminishing
lateral margin despite the clean result.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-21-distance-0160-*.json`.

### 23 — Distance per processed frame 0.0160 → 0.0165 m

Ten-lap results: Waveshare **2.005**, technical **2.055**, hairpin **2.003
m/s**; zero off-road events; macro approximately **2.021 m/s** (+2.9% over
experiment 22 and about +84% over the controlled baseline). Mean center
deviation was 0.0306, 0.0229, and 0.0373 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-22-distance-0165-*.json`.

### 24 — Distance per processed frame 0.0165 → 0.0170 m

Ten-lap results: Waveshare **2.059**, technical **2.114**, hairpin **2.058
m/s**; zero off-road events; macro approximately **2.077 m/s** (+2.8% over
experiment 23 and about +89% over the controlled baseline). Mean center
deviation was 0.0312, 0.0250, and 0.0377 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-23-distance-0170-*.json`.

### 25 — Distance per processed frame 0.0170 → 0.0175 m

Ten-lap results: Waveshare **2.113**, technical **2.173**, hairpin **2.112
m/s**; zero off-road events; macro approximately **2.133 m/s** (+2.7% over
experiment 24 and about +94% over the controlled baseline). Mean center
deviation was 0.0321, 0.0286, and 0.0381 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-24-distance-0175-*.json`.

### 26 — Distance per processed frame 0.0175 → 0.0180 m

Ten-lap results: Waveshare **2.167**, technical **2.231**, hairpin **2.165
m/s**; zero off-road events; macro approximately **2.188 m/s** (+2.6% over
experiment 25 and about +99% over the controlled baseline). Mean center
deviation was 0.0337, 0.0294, and 0.0378 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-25-distance-0180-*.json`.

### 27 — Distance per processed frame 0.0180 → 0.0185 m

Ten-lap results: Waveshare **2.219**, technical **2.289**, hairpin **2.219
m/s**; zero off-road events; macro approximately **2.242 m/s** (+2.5% over
experiment 26 and about +104% over the controlled baseline). Mean center
deviation was 0.0354, 0.0313, and 0.0378 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-26-distance-0185-*.json`.

### 28 — Distance per processed frame 0.0185 → 0.0190 m

Ten-lap results: Waveshare **2.272**, technical **2.347**, hairpin **2.271
m/s**; zero off-road events; macro approximately **2.297 m/s** (+2.5% over
experiment 27 and about +109% over the controlled baseline). Mean center
deviation was 0.0375, 0.0349, and 0.0368 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-27-distance-0190-*.json`.

### 29 — Distance per processed frame 0.0190 → 0.0195 m

Ten-lap results: Waveshare **2.323**, technical **2.404**, hairpin **2.323
m/s**; zero off-road events; macro approximately **2.350 m/s** (+2.3% over
experiment 28 and about +114% over the controlled baseline). The governor
ceiling is now 2.496 m/s at the fixed 128 FPS, effectively matching the 2.5
m/s requested-speed cap. Mean center deviation was 0.0384, 0.0371, and 0.0380
m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-28-distance-0195-*.json`.

### 30 — Governor acceleration 0.8 → 1.0 m/s²

Hypothesis: once the 2.5 m/s ceiling is reachable, a faster but still
physically plausible launch ramp raises lap-average speed without changing
steady-state lateral demand.

Ten-lap results: Waveshare **2.354**, technical **2.420**, hairpin **2.353
m/s**; zero off-road events; macro approximately **2.376 m/s** (+1.1% over
experiment 29). Mean center deviation was 0.0391, 0.0376, and 0.0379 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-29-accel-10-*.json`.

### 31 — Governor acceleration 1.0 → 1.2 m/s²

Ten-lap results: Waveshare **2.374**, technical **2.432**, hairpin **2.374
m/s**; zero off-road events; macro approximately **2.393 m/s** (+0.7% over
experiment 30). Mean center deviation was 0.0384, 0.0373, and 0.0383 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-30-accel-12-*.json`.

### 32 — Governor acceleration 1.2 → 1.4 m/s²

Result: **rejected**. The hairpin recorded one off-road event in 10 laps;
recovery reduced mean speed to 2.290 m/s. The strict gate stopped the other
tracks. This isolates a launch/ramp stability boundary because steady-state
speed is unchanged from experiment 31.

Evidence: `build/benchmarks/race-opt-fixed-31-accel-14-tight_hairpin.json`.

### 33 — Governor acceleration 1.2 → 1.3 m/s²

Ten-lap results: Waveshare **2.381**, technical **2.436**, hairpin **2.382
m/s**; zero off-road events; macro approximately **2.400 m/s** (+0.3% over
experiment 31). Mean center deviation was 0.0387, 0.0382, and 0.0382 m.
Waveshare remained safe with actual Core ML completion at 118.7 FPS.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-32-accel-13-*.json`.

### 34 — Governor acceleration 1.3 → 1.35 m/s²

Ten-lap results: Waveshare **2.385**, technical **2.438**, hairpin **2.385
m/s**; zero off-road events; macro approximately **2.403 m/s** (+0.1% over
experiment 33). Mean center deviation was 0.0389, 0.0371, and 0.0396 m.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-33-accel-135-*.json`.

### 35 — Governor acceleration 1.35 → 1.375 m/s²

Ten-lap results: Waveshare **2.387**, technical **2.439**, hairpin **2.387
m/s**; zero off-road events; macro approximately **2.404 m/s** (+0.1% over
experiment 34). Mean center deviation was 0.0392, 0.0381, and 0.0397 m.
The approximately 0.002 m/s gain shows that this parameter is effectively
exhausted near the rejected 1.4 m/s² boundary.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-34-accel-1375-*.json`.

### 36 — Lookahead 0.12 → 0.10 s with 1.4 m/s² acceleration

Hypothesis: earlier turn-in could compensate for the single hairpin exit seen
at the faster launch ramp.

Result: **rejected**. The hairpin again recorded one off-road event in 10
laps; recovery reduced mean speed to 2.291 m/s. Shorter lookahead did not move
the 1.4 m/s² launch boundary, so both candidate settings were reverted.

Evidence:
`build/benchmarks/race-opt-fixed-35-lookahead-010-accel-14-tight_hairpin.json`.

### 37 — Stanley controller with 1.4 m/s² acceleration

Hypothesis: Stanley's explicit heading and cross-track terms might tolerate
the faster ramp better than pure pursuit.

Result: **rejected**. The configured Stanley controller recorded 29 hairpin
exits in 10 laps and only 1.120 m/s mean speed. Its existing tuning is not
competitive in this high-speed, vision-derived-path regime. Pure pursuit and
1.375 m/s² were restored.

Evidence: `build/benchmarks/race-opt-fixed-36-stanley-accel-14-tight_hairpin.json`.

### 38 — Off-road event diagnostics and 1.4 m/s² rerun

Material tooling change: benchmark results now record each off-road event's
accumulated progress, lap, and simulation time. Existing metrics and control
behaviour are unchanged.

A repeat of minimum-time/pure-pursuit at 1.4 m/s² completed 10 laps and then
30 laps with zero exits, contradicting experiment 32's single 10-lap exit.
The setting remains **unaccepted** because the campaign has observed a real
incident and the asynchronous actual-perception path can expose rare timing
variations.

Evidence: `build/benchmarks/race-opt-diagnostic-accel-14-offroad-location-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-accel-14-30lap-tight_hairpin.json`.

### 39 — Centerline planner with 1.4 m/s² acceleration

Result: **rejected**. The hairpin recorded two exits in 10 laps and 2.198 m/s
mean speed. Removing the small minimum-time offset reduced corner margin.

Evidence: `build/benchmarks/race-opt-fixed-37-centerline-accel-14-tight_hairpin.json`.

### 40 — Local racing line with 1.4 m/s² acceleration

The 10-lap development matrix initially passed and narrowly improved every
mean: Waveshare 2.389, technical 2.440, hairpin 2.388 m/s, all zero exits.
Technical mean deviation also fell from 0.0381 to 0.0281 m.

Result after extended confirmation: **rejected**. The 30-lap hairpin recorded
two exits, at laps 0.680 and 19.816. Strict safety therefore outweighs the
marginal speed gain.

Evidence: `build/benchmarks/race-opt-fixed-38-local-racing-line-accel-14-*.json`
and `build/benchmarks/race-opt-robustness-local-racing-line-accel-14-30lap-tight_hairpin.json`.

### 41 — Incumbent 30-lap robustness confirmation

The restored minimum-time line, pure pursuit, and 1.375 m/s² acceleration
completed 30 hairpin laps at **2.459 m/s** mean with zero exits. This confirms
the accepted configuration under the same extended gate that rejected the
local planner.

Evidence: `build/benchmarks/race-opt-robustness-incumbent-accel-1375-30lap-tight_hairpin.json`.

### 42 — Minimum-time offset 0.03 → 0.04 m with 1.4 m/s² acceleration

Hypothesis: a controlled midpoint between the stable 0.03 m minimum-time line
and unstable wide local line can add corner margin while unlocking the faster
ramp.

Ten-lap results: Waveshare **2.3886**, technical **2.4393**, hairpin **2.3884
m/s**; zero off-road events. Each mean strictly improved over experiment 35
by 0.0007–0.0016 m/s. The hairpin then completed 30 laps at **2.459 m/s**
with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-39-min-time-offset-004-accel-14-*.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-004-accel-14-30lap-tight_hairpin.json`.

### 43 — Governor acceleration 1.4 → 1.45 m/s²

Ten-lap results: Waveshare **2.392**, technical **2.441**, hairpin **2.392
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.460
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-40-min-time-offset-004-accel-145-*.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-004-accel-145-30lap-tight_hairpin.json`.

### 44 — Governor acceleration 1.45 → 1.50 m/s²

Ten-lap results: Waveshare **2.395**, technical **2.442**, hairpin **2.395
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.461
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-41-min-time-offset-004-accel-15-*.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-004-accel-15-30lap-tight_hairpin.json`.

### 45 — Governor acceleration 1.50 → 1.60 m/s²

Ten-lap results: Waveshare **2.400**, technical **2.445**, hairpin **2.400
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.463
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-42-min-time-offset-004-accel-16-*.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-004-accel-16-30lap-tight_hairpin.json`.

### 46 — Governor acceleration 1.60 → 2.00 m/s²

Ten-lap results: Waveshare **2.416**, technical **2.454**, hairpin **2.416
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.469
m/s** with zero exits. This matches the simulator's configured physical
acceleration rate, so the ramp search stops here.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-43-min-time-offset-004-accel-20-*.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-004-accel-20-30lap-tight_hairpin.json`.

### 47 — Exact 2.5 m/s governor ceiling

Changed the distance budget from 0.0195 to the analytically derived
**0.01953125 m/frame**, because 128 FPS × 0.01953125 m = exactly 2.5 m/s.
This removes a 0.004 m/s under-run without exceeding the requested cap.

Ten-lap results: Waveshare **2.419**, technical **2.458**, hairpin **2.419
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.472
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-44-exact-speed-cap-*.json` and
`build/benchmarks/race-opt-robustness-exact-speed-cap-30lap-tight_hairpin.json`.

### 48 — Governor acceleration 2.0 → 2.5 m/s²

The 10-lap hairpin initially passed at **2.432 m/s**, but the 30-lap run
recorded one exit at lap **14.820**. Because the event occurred long after the
launch transient, it exposes rare asynchronous perception/path sensitivity
rather than a repeatable acceleration-only failure.

Status: **rejected** under the strict zero-incident gate.

Evidence: `build/benchmarks/race-opt-fixed-45-governor-accel-25-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-governor-accel-25-30lap-tight_hairpin.json`.

### 49 — Minimum-time offset 0.04 → 0.05 m at 2.5 m/s²

The 10-lap hairpin passed, but the 30-lap confirmation recorded two exits at
laps **10.819** and **14.303**. More lateral freedom did not absorb the rare
perception/path excursions and was worse than 0.04 m.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-fixed-46-min-time-offset-005-accel-25-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-min-time-offset-005-accel-25-30lap-tight_hairpin.json`.

### 50 — Temporal path filter at 2.5 m/s²

The filter passed the 10-lap hairpin at 2.434 m/s, but the 30-lap run
recorded **seven exits** and fell to 2.331 m/s mean. Even its 5 ms time
constant adds harmful phase lag at this speed, confirming the earlier choice
to keep temporal filtering available but disabled.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-fixed-47-temporal-accel-25-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-temporal-accel-25-30lap-tight_hairpin.json`.

### 51 — Steering-rate limit 4.0 → 5.0 rad/s at 2.5 m/s²

Hypothesis: the 4 rad/s limiter, which clean runs repeatedly reached exactly,
delayed corrective turn-in after skipped perception frames.

Ten-lap results: Waveshare **2.432**, technical **2.464**, hairpin **2.432
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.477
m/s** with zero exits. This safely unlocks the faster governor ramp rejected
in experiment 48.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-48-steering-rate-50-accel-25-*.json`
and `build/benchmarks/race-opt-robustness-steering-rate-50-accel-25-30lap-tight_hairpin.json`.

### 52 — Governor acceleration 2.5 → 3.0 m/s²

Ten-lap results: Waveshare **2.441**, technical **2.469**, hairpin **2.441
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.480
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-49-steering-rate-50-accel-30-*.json`
and `build/benchmarks/race-opt-robustness-steering-rate-50-accel-30-30lap-tight_hairpin.json`.

### 53 — Governor acceleration 3.0 → 4.0 m/s²

Ten-lap results: Waveshare **2.452**, technical **2.475**, hairpin **2.452
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.484
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-50-steering-rate-50-accel-40-*.json`
and `build/benchmarks/race-opt-robustness-steering-rate-50-accel-40-30lap-tight_hairpin.json`.

### 54 — Governor acceleration 4.0 → 6.0 m/s²

The 10-lap hairpin passed at 2.463 m/s, but the 30-lap confirmation recorded
one exit at lap **25.800**. The event was long after ramp completion, again
showing the rare-event boundary at sustained 2.5 m/s.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-fixed-51-steering-rate-50-accel-60-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-steering-rate-50-accel-60-30lap-tight_hairpin.json`.

### 55 — Governor acceleration 5.0 m/s² midpoint

Result: **rejected**. The hairpin recorded one exit within the 10-lap gate
and mean speed fell to 2.418 m/s after recovery. No other tracks were run.

Evidence: `build/benchmarks/race-opt-fixed-52-steering-rate-50-accel-50-tight_hairpin.json`.

### 56 — Matched steering-rate/acceleration increase to 6/6

Hypothesis: scale lateral recovery authority with the sharper longitudinal
target ramp, after 6 m/s² failed with the 5 rad/s steering ceiling.

Ten-lap results: Waveshare **2.463**, technical **2.481**, hairpin **2.463
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.488
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-53-steering-rate-60-accel-60-*.json`
and `build/benchmarks/race-opt-robustness-steering-rate-60-accel-60-30lap-tight_hairpin.json`.

### 57 — Matched steering-rate/acceleration increase to 8/8

Ten-lap results: Waveshare **2.468**, technical **2.484**, hairpin **2.468
m/s**; zero off-road events. The hairpin also completed 30 laps at **2.489
m/s** with zero exits.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-fixed-54-steering-rate-80-accel-80-*.json`
and `build/benchmarks/race-opt-robustness-steering-rate-80-accel-80-30lap-tight_hairpin.json`.

### 58 — Matched steering-rate/acceleration increase to 12/12

The 10-lap hairpin passed at 2.474 m/s, but the 30-lap run recorded two exits
at laps **5.654** and **12.691**.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-fixed-55-steering-rate-120-accel-120-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-steering-rate-120-accel-120-30lap-tight_hairpin.json`.

### 59 — Matched steering-rate/acceleration midpoint 10/10

Result: **rejected**. The hairpin recorded two exits within 10 laps and mean
speed fell to 2.417 m/s after recovery.

Evidence: `build/benchmarks/race-opt-fixed-56-steering-rate-100-accel-100-tight_hairpin.json`.

### 60 — Matched steering-rate/acceleration midpoint 9/9

Result: **rejected**. The hairpin recorded three exits within 10 laps and
mean speed fell to 2.385 m/s. The paired ramp/rate search therefore ends at
the verified 8/8 incumbent.

Evidence: `build/benchmarks/race-opt-fixed-57-steering-rate-90-accel-90-tight_hairpin.json`.

### 61 — Innovation gate, 0.06 m median threshold

Material feature: added an optional hard path-innovation gate that propagates
the last accepted path but does not smooth valid observations. It rejects an
isolated whole-path lateral jump and accepts persistent changes after two
rejections. Thresholds and counters are configurable/reported.

The 10-lap hairpin passed at 2.468 m/s, but the 30-lap run recorded **five
exits**. The 0.06 m threshold rejected legitimate fast path changes and
introduced effective lag.

Status: **rejected; threshold retained only as experiment evidence**.

Reports: `build/benchmarks/race-opt-fixed-innovation-gate-tight_hairpin.json`
and `build/benchmarks/race-opt-robustness-innovation-gate-30lap-tight_hairpin.json`.

### 62 — Innovation gate threshold 0.06 → 0.12 m

Result: **rejected**. The 30-lap hairpin recorded three exits. The gate was
still rejecting path updates, so this threshold continued to classify valid
high-speed geometry changes as outliers.

Evidence: `build/benchmarks/race-opt-innovation-gate-threshold-012-30lap-tight_hairpin.json`.

### 63 — Innovation gate threshold 0.12 → 0.20 m

Result: **rejected**. The 30-lap hairpin recorded four exits. A hard
whole-path median gate cannot distinguish the valid rapid geometry changes
from harmful observations in this camera/controller representation.

Decision: remove the experimental gate from production code and retain the
reports as evidence. Temporal filtering remains optional and disabled.

Evidence: `build/benchmarks/race-opt-innovation-gate-threshold-020-30lap-tight_hairpin.json`.

### 64 — Deterministic actual-model perception scheduling

The prior long-run failures were not repeatable at the same configuration.
The cause was the asynchronous Core ML worker: wall-clock completion timing
changed which simulated camera frames reached the controller, so identical
runs did not exercise identical trajectories.

Added an opt-in deterministic benchmark mode. It still executes the actual
SegFormer Core ML model, but schedules completed masks at a fixed **128 Hz**
in simulated time and selects frames from the **200 Hz** simulated camera.
Two identical 10-lap hairpin runs now match exactly on all trajectory and
control metrics: **2.469855 m/s**, **0 exits**, mean deviation **0.039745 m**,
and 2,933 submitted/completed masks with no replacements.

Status: **accepted as the optimization protocol**. This is a measurement fix,
not a performance claim. Wall-clock model throughput remains reported
separately (about 112 fps during these thermally sustained runs).

Reports: `build/benchmarks/race-opt-deterministic-00-tight_hairpin.json` and
`build/benchmarks/race-opt-deterministic-00-repeat-tight_hairpin.json`.

### 65 — Deterministic incumbent confirmation at 8/8

Ten-lap results with the 8 m/s² acceleration and 8 rad/s steering-rate
incumbent: Waveshare **2.470**, technical chicane **2.484**, and tight hairpin
**2.470 m/s**; all tracks completed with **zero off-road events**. The
30-lap hairpin confirmation also passed with zero exits at **2.490 m/s**.

Status: **confirmed incumbent under the deterministic protocol**.

Reports: `build/benchmarks/race-opt-deterministic-final-*.json` and
`build/benchmarks/race-opt-deterministic-final-30lap-tight_hairpin.json`.

### 66 — Deterministic matched steering-rate/acceleration increase to 10/10

The short matrix improved on every track with zero exits: Waveshare
**2.4735**, technical chicane **2.4862**, and tight hairpin **2.4733 m/s**.
However, the 30-lap hairpin gate recorded one deterministic exit at lap
**14.678**. The faster target ramp therefore consumes long-run lateral margin
that a 10-lap sample does not reveal.

Status: **rejected** despite the short-run speed gain.

Reports: `build/benchmarks/race-opt-deterministic-01-rate-10-accel-10-*.json`.

### 67 — Deterministic 9/9 midpoint

The 9 m/s² acceleration and 9 rad/s steering-rate midpoint improved all
three 10-lap means with zero exits: Waveshare **2.4720**, technical chicane
**2.4854**, and tight hairpin **2.4717 m/s**. The 30-lap hairpin also passed
with zero exits at **2.4905 m/s**, versus 2.4899 m/s for the 8/8 incumbent.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-02-rate-9-accel-9-*.json`.

### 68 — Deterministic 9.5/9.5 boundary probe

The 30-lap hairpin recorded one exit, so the paired rate/ramp boundary lies
between the accepted 9/9 setting and 9.5/9.5. The short matrix was skipped
after the durability gate failed.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-03-rate-9p5-accel-9p5-30lap-tight_hairpin.json`.

### 69 — Deterministic 9.25/9.25 boundary probe

The 30-lap hairpin again recorded one exit. The safe boundary is now bounded
to 9.0–9.25, and the full matrix was skipped.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-04-rate-9p25-accel-9p25-30lap-tight_hairpin.json`.

### 70 — Deterministic 9.125/9.125 boundary probe

The 30-lap hairpin passed with zero exits at **2.4907 m/s**. All three
10-lap means also improved, though only marginally: Waveshare **2.47216**,
technical chicane **2.48545**, and tight hairpin **2.47193 m/s**. No track
recorded an off-road event.

Status: **accepted; new incumbent**. Because the smallest gain is only about
0.00004 m/s, one final midpoint is justified before ending this search axis.

Reports: `build/benchmarks/race-opt-deterministic-05-rate-9p125-accel-9p125-*.json`.

### 71 — Deterministic 9.1875/9.1875 boundary probe

The 30-lap hairpin recorded three exits. This is a clear regression rather
than a marginal boundary result, so the paired rate/ramp search ends at the
accepted 9.125/9.125 setting.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-06-rate-9p1875-accel-9p1875-30lap-tight_hairpin.json`.

### 72 — Racing-line offset 0.040 → 0.045 m at 9.25/9.25

The wider line recorded one exit in the 30-lap hairpin and increased mean
deviation to 0.0411 m. The curvature benefit did not compensate for reduced
road-edge clearance.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-deterministic-07-offset-0p045-rate-9p25-accel-9p25-30lap-tight_hairpin.json`.

### 73 — Lateral-error gain 0.25 → 0.35 at 9.25/9.25

The 30-lap hairpin recorded two exits. Extra near-field feedback amplifies
short-range path variation and does not improve the dominant pure-pursuit
lookahead term.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-deterministic-08-lateral-gain-0p35-rate-9p25-accel-9p25-30lap-tight_hairpin.json`.

### 74 — Decoupled 9.25 acceleration / 9.125 steering rate

The 30-lap hairpin recorded two exits. This isolates acceleration as the
unsafe part of the paired 9.25/9.25 change; retaining the proven steering
rate does not make the faster launch safe.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-09-rate-9p125-accel-9p25-30lap-tight_hairpin.json`.

### 75 — Steering smoothing 0.035 → 0.025 s at 9.25/9.25

The lower-lag command recorded two exits and increased mean deviation to
0.0419 m. The actuator already has its own response dynamics; removing more
controller smoothing exposes mask-to-mask command variation.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-deterministic-10-smoothing-0p025-rate-9p25-accel-9p25-30lap-tight_hairpin.json`.

### 76 — Interpolated pure-pursuit lookahead targets

Material prototype: linearly interpolate between adjacent spatial path
samples instead of selecting the nearest scanline. This removes target
quantization without adding temporal lag.

The 30-lap hairpin nevertheless recorded one exit at the accepted 9.125
rate/ramp. Nearest-sample selection was providing useful spatial
regularization when adjacent projected samples disagree.

Status: **rejected; prototype removed**.

Evidence: `build/benchmarks/race-opt-deterministic-11-interpolated-lookahead-30lap-tight_hairpin.json`.

### 77 — Pure-pursuit gain 1.75 → 1.80 at 9.25/9.25

The 30-lap hairpin recorded one exit and mean deviation increased to 0.0411
m. Stronger preview curvature authority did not recover the launch margin.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-deterministic-12-pure-pursuit-gain-1p80-rate-9p25-accel-9p25-30lap-tight_hairpin.json`.

### 78 — Minimum-time lateral candidates 7 → 9 at 9.25/9.25

Finer lateral sampling reduces the racing-line offset step from about 13.3
mm to 10 mm. It made the previously unsafe 9.25 ramp pass 30 hairpin laps at
**2.49063 m/s** with zero exits. Every 10-lap mean also improved with zero
exits: Waveshare **2.47231**, technical chicane **2.48555**, and tight
hairpin **2.47215 m/s**.

The chicane mean deviation increased from 0.0365 to 0.0417 m, while hairpin
mean deviation improved slightly. The primary speed/zero-incident gate is
satisfied on all tracks.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-13-racing-line-candidates-9-rate-9p25-accel-9p25-*.json`.

### 79 — Nine-candidate racing line at 9.5/9.5

The 30-lap hairpin recorded two exits. Finer path sampling moves the safe
launch boundary upward, but does not make 9.5 safe.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-14-racing-line-candidates-9-rate-9p5-accel-9p5-30lap-tight_hairpin.json`.

### 80 — Nine-candidate racing line at 9.375/9.375

The 30-lap hairpin passed with zero exits at **2.49070 m/s**. Every 10-lap
mean improved with zero exits: Waveshare **2.47263**, technical chicane
**2.48562**, and tight hairpin **2.47236 m/s**. Hairpin and Waveshare mean
deviation improved; chicane mean deviation increased by about 3.1 mm.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-15-racing-line-candidates-9-rate-9p375-accel-9p375-*.json`.

### 81 — Nine-candidate racing line at 9.4375/9.4375

The 30-lap hairpin passed with zero exits, and the complex-track 10-lap
means improved. Waveshare, however, decreased from **2.472629** to
**2.472617 m/s**. The difference is tiny but violates the predeclared rule
that every track must improve.

Status: **rejected and reverted**.

Reports: `build/benchmarks/race-opt-deterministic-16-racing-line-candidates-9-rate-9p4375-accel-9p4375-*.json`.

### 82 — Minimum-time lateral candidates 9 → 11

The 30-lap hairpin passed, but all three 10-lap means decreased slightly:
Waveshare **2.47257**, technical chicane **2.48557**, and tight hairpin
**2.47235 m/s**. More lateral states added path-choice sensitivity without a
speed benefit.

Status: **rejected and reverted to 9 candidates**.

Reports: `build/benchmarks/race-opt-deterministic-17-racing-line-candidates-11-rate-9p375-accel-9p375-*.json`.

### 83 — Minimum-time forward resampling 14 → 16 stations

The 30-lap hairpin recorded one exit at the accepted rate/ramp. Denser
forward planning overfit local corridor variation instead of improving the
line.

Status: **rejected and reverted**.

Evidence: `build/benchmarks/race-opt-deterministic-18-racing-line-resample-16-candidates-9-rate-9p375-accel-9p375-30lap-tight_hairpin.json`.

### 84 — Road scanline stride 3 → 2 pixels

The 30-lap hairpin passed, shortened completion time by 0.255 s, and reduced
mean and peak deviation. Mean scalar speed nevertheless decreased from
**2.490703** to **2.490682 m/s**, so the candidate fails the requested speed
metric even though it improves lap time and accuracy.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-19-road-row-stride-2-30lap-tight_hairpin.json`.

### 85 — Road scanline stride 3 → 4 pixels

The coarser extraction recorded one exit in the 30-lap hairpin. Reduced
scanline noise does not compensate for lost road geometry.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-20-road-row-stride-4-30lap-tight_hairpin.json`.

### 86 — Two-row extraction plus 9.5/9.5 rate/ramp

The denser path alone created lateral margin but did not improve mean speed.
Converting that margin into a faster launch did: all three 10-lap means
improved with zero exits—Waveshare **2.47264**, technical chicane **2.48578**,
and tight hairpin **2.47245 m/s**. The 30-lap hairpin passed at **2.49074
m/s**. Mean deviation improved on both complex tracks.

Wall-clock benchmark throughput decreased by roughly 5–6%, so controller
stage latency is profiled separately; simulated perception scheduling remains
fixed at 128 Hz from a 200 Hz camera.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-21-road-row-stride-2-rate-9p5-accel-9p5-*.json`.

Controller profiling on a one-lap actual-model run: steering-pipeline mean
**0.94 ms**, p99 **2.06 ms**, maximum **2.25 ms**. This fits the 5 ms period
of the 200 Hz control loop. SegFormer inference averaged 7.96 ms (~125.6
fps), consistent with the fixed 128 Hz simulated-time schedule.

Profile: `build/benchmarks/race-opt-deterministic-21-road-row-stride-2-profiled-tight_hairpin.json`.

### 87 — Two-row extraction at 9.75/9.75

The 30-lap hairpin passed at **2.49087 m/s** with zero exits. All three
10-lap means improved with zero exits: Waveshare **2.47298**, technical
chicane **2.48600**, and tight hairpin **2.47280 m/s**. Technical-chicane
peak deviation improved by about 15 mm; Waveshare mean deviation increased
by 0.62 mm.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-22-road-row-stride-2-rate-9p75-accel-9p75-*.json`.

### 88 — Two-row extraction at 10/10

The 30-lap hairpin recorded one exit. Stride 2 raises the safe launch
boundary substantially, but does not make 10/10 safe.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-23-road-row-stride-2-rate-10-accel-10-30lap-tight_hairpin.json`.

### 89 — Two-row extraction at 9.875/9.875

The 30-lap hairpin passed at **2.49092 m/s** with zero exits. Every 10-lap
mean improved with zero exits: Waveshare **2.47319**, technical chicane
**2.48607**, and tight hairpin **2.47297 m/s**. Mean deviation increased by
0.10–0.62 mm; chicane peak deviation increased by 13.8 mm.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-24-road-row-stride-2-rate-9p875-accel-9p875-*.json`.

### 90 — Full-row extraction at 10/10

The 30-lap hairpin passed at **2.49102 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47340**, technical chicane **2.48622**, and
tight hairpin **2.47323 m/s**. Mean and peak deviations improved on every
track relative to stride 2 at 9.875/9.875.

The steering pipeline remains within the 200 Hz control period: mean **1.65
ms**, p99 **3.05 ms**, maximum **3.10 ms**. Segmentation runs asynchronously
in deployment, so its ~7.93 ms inference latency is not part of the steering
stage's 5 ms deadline.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-25-road-row-stride-1-rate-10-accel-10-*.json` and
`build/benchmarks/race-opt-deterministic-25-road-row-stride-1-profiled-tight_hairpin.json`.

### 91 — Full-row extraction at 10.5/10.5

The 30-lap hairpin passed at **2.49123 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47408**, technical chicane **2.48653**, and
tight hairpin **2.47389 m/s**. Mean deviations increased by 0.07–1.76 mm,
with no boundary event.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-26-road-row-stride-1-rate-10p5-accel-10p5-*.json`.

### 92 — Full-row extraction at 11/11

The 30-lap hairpin passed at **2.49142 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47466**, technical chicane **2.48680**, and
tight hairpin **2.47442 m/s**. Waveshare and hairpin deviation improved;
chicane mean deviation increased by 1.12 mm.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-27-road-row-stride-1-rate-11-accel-11-*.json`.

### 93 — Full-row extraction at 12/12

The 30-lap hairpin passed at **2.49178 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47572**, technical chicane **2.48734**, and
tight hairpin **2.47551 m/s**. Mean deviation improved on every track,
although peak deviation increased.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-28-road-row-stride-1-rate-12-accel-12-*.json`.

### 94 — Full-row extraction at 13/13

The 30-lap hairpin passed at **2.49207 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47655**, technical chicane **2.48781**, and
tight hairpin **2.47635 m/s**. Peak deviation improved on all three tracks.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-29-road-row-stride-1-rate-13-accel-13-*.json`.

### 95 — Full-row extraction at 14/14

The 30-lap hairpin passed at **2.49232 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47726**, technical chicane **2.48817**, and
tight hairpin **2.47712 m/s**. Hairpin mean and peak deviation improved;
Waveshare and chicane mean deviation increased.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-30-road-row-stride-1-rate-14-accel-14-*.json`.

### 96 — Full-row extraction at 16/16

The 30-lap hairpin passed at **2.49273 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.47848**, technical chicane **2.48881**, and
tight hairpin **2.47834 m/s**. Hairpin mean/peak deviation increased;
technical mean deviation improved slightly.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-31-road-row-stride-1-rate-16-accel-16-*.json`.

### 97 — Full-row extraction at 20/20

The 30-lap hairpin passed at **2.49330 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.48017**, technical chicane **2.48967**, and
tight hairpin **2.48003 m/s**. Mean deviation improved on every track;
chicane peak deviation increased by about 11.4 mm.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-32-road-row-stride-1-rate-20-accel-20-*.json`.

### 98 — Full-row extraction at 30/30

The 30-lap hairpin passed at **2.49406 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.48243**, technical chicane **2.49086**, and
tight hairpin **2.48227 m/s**. Peak deviation improved markedly on both
complex tracks. Actual peak steering-command rate was 27.65 rad/s, so the
30 rad/s ceiling is no longer binding.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-33-road-row-stride-1-rate-30-accel-30-*.json`.

### 99 — Acceleration 50 with steering rate held at 30

The 30-lap hairpin recorded one exit. With steering no longer rate-limited,
this isolates the failure to the faster longitudinal launch/phase.

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-34-road-row-stride-1-rate-30-accel-50-30lap-tight_hairpin.json`.

### 100 — Acceleration 40 with steering rate held at 30

The 30-lap hairpin passed at **2.49445 m/s**. Every 10-lap mean improved
with zero exits: Waveshare **2.48353**, technical chicane **2.49147**, and
tight hairpin **2.48342 m/s**. Hairpin/Waveshare peak deviation improved;
chicane mean and peak deviation increased.

Status: **accepted; new incumbent**.

Reports: `build/benchmarks/race-opt-deterministic-35-road-row-stride-1-rate-30-accel-40-*.json`.

### 101 — Acceleration 45 with steering rate held at 30

The 30-lap hairpin recorded one exit, narrowing the safe simulated launch
boundary to 40–45 m/s².

Status: **rejected**.

Evidence: `build/benchmarks/race-opt-deterministic-36-road-row-stride-1-rate-30-accel-45-30lap-tight_hairpin.json`.

### 102 — Acceleration 42.5 with steering rate held at 30

The 30-lap hairpin recorded one exit. The safe simulated launch setting
remains 40 m/s² under this protocol; further midpoint refinement is not
worth the diminishing speed gain.

Status: **rejected and reverted to acceleration 40 / steering rate 30**.

Evidence: `build/benchmarks/race-opt-deterministic-37-road-row-stride-1-rate-30-accel-42p5-30lap-tight_hairpin.json`.

### 103 — Final 30-lap three-track robustness matrix

The accepted full-row, nine-candidate, pure-pursuit configuration with
acceleration 40 m/s² and steering-command rate 30 rad/s completed 30 laps on
all target tracks with **zero off-road events**:

- Waveshare 3×2: **2.494 m/s**, mean deviation 0.0407 m.
- Technical chicane: **2.497 m/s**, mean deviation 0.0389 m.
- Tight hairpin: **2.494 m/s**, mean deviation 0.0391 m.

Status: **final robustness gate passed**.

Reports: `build/benchmarks/race-opt-deterministic-final-30lap-waveshare_3x2.json`,
`build/benchmarks/race-opt-deterministic-final-30lap-technical_chicane.json`, and
`build/benchmarks/race-opt-deterministic-35-road-row-stride-1-rate-30-accel-40-30lap-tight_hairpin.json`.

### 104 — Non-race pedestrian regression repair

Package synchronization exposed that the pedestrian-avoidance regression
cleared its object by only 22 mm against a configured 50 mm criterion.
Offsets of 90–100 mm caused an off-road recovery and collision; 80–82.5 mm
were safe but missed clearance. A configurable **84 mm** avoidance offset
produced **52.8 mm** clearance with zero collisions and zero off-road events.

This setting is inactive in all race benchmarks requested for this campaign.
The complete driving-benchmark regression passes after the change.

### 105 — Current asynchronous deployment certificate and complete matrix

Hypothesis: the deterministic 200 Hz-camera protocol establishes controller
quality, but the deployable limit should also survive the live asynchronous
inference scheduler and every selectable model/controller/path tuple must have
current certification evidence.

The live search certified the incumbent SegFormer-B0 Core ML FP16 384 px,
pure-pursuit, minimum-time-line tuple at **2.425 m/s simulated** with zero
off-road events. Applying the configured 0.80 simulated-to-real transfer
factor gives a provisional deployment limit of **1.940 m/s**. A 2.450 m/s
candidate recorded one tight-hairpin exit, so the search boundary was
exercised rather than merely reaching the configured maximum.

The complete 30-case matrix then produced **1 certified / 29 explicitly
uncertified / 0 execution errors**. The incumbent was the sole certified
tuple. Notably, the 256 px Core ML model failed every control/path combination
at the 0.5 m/s minimum, while the 512 px model also failed to produce a safe
tuple; extra or reduced spatial resolution did not improve closed-loop safety
on the textured simulator scenes.

Certification coverage is now **30/30**, with zero missing and zero stale
cases (three TensorRT/B1 artifacts are recorded as unavailable on this Mac).
All **41/41** CTest targets pass.

Status: **validation accepted; race incumbent unchanged**. The deterministic
race result remains 2.484–2.491 m/s over ten laps and 2.494–2.497 m/s over the
30-lap robustness runs, with zero exits on all three target tracks.

Evidence: `build/benchmarks/current-speed-certificate-2026-08-03.json`,
`build/benchmarks/full-speed-matrix/summary.json`,
`benchmarks/speed_certification_results.json`, and
`benchmarks/certified_speed_limits.json`.

### 106 — Acceleration-boundary refinement at 41.25 m/s²

Hypothesis: 41.25 m/s² may retain the small mean-speed gain of the rejected
42.5 m/s² candidate while restoring the 30-lap hairpin safety margin.

Result: **rejected and reverted to 40 m/s²**. The candidate recorded two
off-road events on hairpin laps 15.70 and 16.65 and averaged only **2.48317
m/s** after recovery penalties. Failures at 41.25, 42.5, 45, and 50 m/s² now
show that this is not a launch-only parameter: it controls re-acceleration
after intermittent perception/governor reductions, and the resulting phase
shift can destabilize later laps. The strict hairpin gate stopped the
experiment before the other tracks.

Evidence: `build/benchmarks/race-opt-deterministic-38-accel-41p25-30lap-tight_hairpin.json`.

### 107 — Moderate steering smoothing with 41.25 m/s² acceleration

Hypothesis: full-row road extraction may now tolerate reducing controller
smoothing from 0.035 to 0.030 s, and the lower steering lag may stabilize the
otherwise unsafe 41.25 m/s² re-acceleration setting.

Result: **rejected and fully reverted**. The candidate recorded two hairpin
exits on laps 13.66 and 27.66 and averaged **2.48313 m/s**. Mean deviation did
improve from 0.03927 to 0.03862 m relative to experiment 106, but the strict
zero-exit criterion still failed. Steering lag is therefore not the dominant
cause of the acceleration boundary; the smaller smoothing constant merely
changed the failure phase.

Evidence: `build/benchmarks/race-opt-deterministic-39-smoothing-0p030-accel-41p25-30lap-tight_hairpin.json`.

### 108 — Acceleration-boundary midpoint at 40.625 m/s²

Hypothesis: the midpoint between the proven 40.0 m/s² incumbent and rejected
41.25 m/s² candidate may recover a small amount of longitudinal response
without crossing the sharp phase-stability boundary.

The 30-lap hairpin gate passed, followed by ten- and 30-lap runs on every
target track. Every exact mean increased with zero off-road events:

| Track | 40.0 m/s², 30 laps | 40.625 m/s², 30 laps | Change | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 2.494477926 | 2.494502952 | +0.000025027 | 0 |
| Technical chicane | 2.497135946 | 2.497138590 | +0.000002643 | 0 |
| Tight hairpin | 2.494445031 | 2.494469019 | +0.000023988 | 0 |

The 30-lap macro mean increased from **2.495352968 to 2.495370187 m/s**.
This is only a 0.00069% gain and the car is already within 0.19% of the
configured 2.5 m/s ceiling, so further boundary subdivision is not a useful
risk/reward trade. The simulation-only acceleration remains capped to 0.8
m/s² by the JetRacer Pro platform profile on real hardware.

Status: **accepted; new incumbent at the practical speed ceiling**.

Reports: `build/benchmarks/race-opt-deterministic-40-accel-40p625-10lap-*.json`
and `build/benchmarks/race-opt-deterministic-40-accel-40p625-30lap-*.json`.

### 109 — Broader deployment regression gate for 40.625 m/s²

The accepted target-track result in experiment 108 changed the runtime
fingerprint, so the complete asynchronous certification matrix was rerun
before finalizing it. The result was **0 certified / 30 explicitly
uncertified / 0 execution errors**. In particular, the incumbent 384 px,
pure-pursuit, minimum-time tuple stayed on-road at the initial 0.5 m/s
candidate but exceeded the open-oval deviation limit; no alternate tuple was
certifiable.

The target-track macro gain was only 0.00069%, while losing the deployable
speed certificate is a material codebase regression. Experiment 108 is
therefore superseded at this broader gate: **40.625 m/s² rejected and reverted
to the fully certified 40.0 m/s² incumbent**. The target-track reports remain
valid boundary evidence but are not promoted to configuration.

Evidence: `build/benchmarks/full-speed-matrix-40p625/summary.json`.

### 110 — Policy-aware certificate cache and 40.625 re-evaluation

Investigation showed experiment 109's apparent regression was not caused by
40.625 m/s². The old 40.0 matrix had skipped its incumbent case because the
registry contained a short **1-lap, 1-trial, 2.4–2.5 m/s** certificate. The
matrix incorrectly treated matching configuration identity as sufficient,
even though its own policy requires **3 laps, 2 trials, 0.5–2.5 m/s**. A
direct full-policy run at 40.0 failed the same 0.5 m/s open-oval deviation
gate with zero exits.

The matrix now reuses a registry certificate only when the certificate
report's complete search policy exactly matches the current matrix policy.
A regression test covers matching, mismatched, and missing reports. Rerunning
the 40.0 matrix with resume replaced the false `already_certified` status:
the honest result is **0 certified / 30 uncertified / 0 errors**, with 30/30
coverage.

Because 40.625 did not introduce the full-policy failure and it improved all
three requested tracks in both ten- and 30-lap zero-exit runs, experiment 109
is superseded. **40.625 m/s² is reinstated as the target-track incumbent**;
the relevant high-speed live certificate is evaluated separately.

Evidence: `build/benchmarks/full-policy-incumbent-40p0.json` and
`build/benchmarks/full-speed-matrix/summary.json`.

### 111 — Relevant high-speed live certificate at 40.625 m/s²

The live asynchronous search was repeated over the operational 2.4–2.5 m/s
band after reinstating 40.625 m/s². Speeds 2.40 and 2.45 passed all four
certificate tracks with zero exits. The 2.50 candidate was not fully
exercised under thermally limited live throughput, and the 2.475 refinement
candidate recorded one tight-hairpin exit.

The new certificate is therefore **2.450 m/s simulated / 1.960 m/s
provisional deployment**, improving the 40.0 certificate from 2.425 / 1.940
m/s. The 40.625 full-policy catalog remains honestly marked 0 certified / 30
uncertified because the minimum-speed search is non-monotonic on the
minimum-time path; coverage is still complete at 30/30.

Status: **accepted at the one-lap gate; later superseded by experiment 118's
ten-lap robust certificate**. The 40.625 race incumbent is unchanged.

Evidence: `build/benchmarks/current-speed-certificate-40p625-2026-08-03.json`
and `build/benchmarks/full-speed-matrix-40p625/summary.json`.

### 112 — Final acceleration midpoint at 40.9375 m/s²

Hypothesis: the midpoint between accepted 40.625 and rejected 41.25 m/s² may
provide another small longitudinal improvement without crossing the hairpin
phase-stability boundary.

Result: **rejected and reverted to 40.625 m/s²**. The candidate recorded two
off-road events in the 30-lap hairpin gate and averaged **2.483 m/s** after
recoveries. The boundary is now bracketed to 40.625–40.9375 m/s²; further
subdivision is not justified given the incumbent's 0.00069% gain and proximity
to the 2.5 m/s ceiling.

Evidence: `build/benchmarks/race-opt-deterministic-41-accel-40p9375-30lap-tight_hairpin.json`.

### 113 — Alternate texture/clutter seed robustness

Hypothesis: the accepted full-row controller and 384 px model should retain
near-ceiling, zero-exit performance when only the renderer seeds change.
Track seeds were temporarily offset to 13200 (Waveshare), 12002 (technical),
and 13003 (hairpin), with geometry and all control parameters unchanged.

Waveshare and technical chicane passed ten laps with zero exits at **2.484**
and **2.491 m/s**. Tight hairpin failed with **9 exits** and mean speed fell
to **2.339 m/s**. Enabling the existing temporal path filter on the same seed
also produced 9 exits; mean deviation improved slightly from 0.0472 to 0.0448
m, but safety did not. This indicates a systematic texture-dependent mask
bias rather than temporal jitter.

Status: **robustness hypothesis rejected; original seeds restored and
temporal filtering remains off**. This does not change the fixed-protocol
incumbent, but it identifies multi-seed segmentation robustness as the next
important limitation. No model training is planned, so mitigation should
focus on off-the-shelf model choice, render preprocessing, confidence-aware
fallback, or an ensemble/geometry consistency check.

Evidence: `build/benchmarks/race-opt-seed-b-10lap-*.json` and
`build/benchmarks/race-opt-seed-b-temporal-10lap-tight_hairpin.json`.

### 114 — Segmentation diagnosis of the failing render seed

A 12-second, 200 Hz RGB+semantic hairpin clip was exported for seed 13003 and
sampled identically across the available B0 Core ML resolutions:

| Model | Road IoU | Precision | Recall | Inference FPS |
|---|---:|---:|---:|---:|
| 256 px FP16 | 0.1280 | 0.1290 | 0.9411 | 121.9 |
| 384 px FP16 | **0.1368** | **0.1392** | 0.8878 | 87.7 |
| 512 px FP16 | 0.1302 | 0.1344 | 0.8051 | 64.5 |

The 384 px incumbent remains the best available accuracy/speed tradeoff;
neither lower nor higher resolution is a viable fix. More importantly, the
same 384 px evaluation on the safe original seed 3003 produced a *lower*
global IoU of **0.1260**. Global IoU/confidence is therefore not predictive of
closed-loop safety here. The texture changes where mask errors perturb the
projected road centre/boundaries, not merely their aggregate pixel count.

Learning: a useful runtime guard must operate on projected path geometry
(boundary completeness, lateral jumps, curvature/heading consistency, and
vehicle-edge clearance), rather than slowing solely on mask confidence or
global segmentation quality.

Evidence: `build/benchmarks/hairpin-seed-13003-segmentation-models.json` and
`build/benchmarks/hairpin-seed-3003-control-segmentation.json`.

### 115 — Curvature-aware fallback on the failing seed

Hypothesis: the existing optional curvature speed planner may convert the
texture-sensitive hairpin failure from repeated recoveries into a safe run,
and a higher lateral-acceleration limit may recover useful speed.

At the configured 0.6 m/s² limit, the planner eliminated all nine exits but
averaged only **1.089 m/s**. Limit probes produced:

| Lateral limit (m/s²) | Mean speed (m/s) | Off-road |
|---:|---:|---:|
| 0.6 | 1.089 | 0 |
| 2.0 | 1.201 | 0 |
| 4.0 | 1.330 | 2 |
| 6.0 | 1.358 | 10 |

The safe 2.0 setting is far slower than the unfiltered run's 2.339 m/s mean
even after nine recovery penalties, while 4.0 and 6.0 violate the zero-exit
gate. Curvature-only slowdown is therefore too conservative to meet the speed
objective and has a sharp failure boundary.

Status: **rejected; curvature limit restored to 0.6 m/s², speed planner remains
off, and hairpin seed restored to 3003**.

Evidence: `build/benchmarks/race-opt-seed-b-curvature-speed-*.json`.

### 116 — Final hot-system deterministic repeat

The accepted configuration was repeated for ten laps per target track after
the full matrix, model evaluations, and robustness probes had heated the Mac.
All 30 laps passed with zero exits: Waveshare **2.483584**, technical chicane
**2.491492**, and tight hairpin **2.483469 m/s**. These reproduce the accepted
ten-lap means exactly while measured Core ML completion throughput was only
88.6–92.4 FPS.

Status: **passed**. This confirms that fixed 128 FPS governor telemetry plus
deterministic actual-model scheduling isolates controller performance from
wall-clock thermal throttling as designed.

Evidence: `build/benchmarks/race-opt-final-hot-repeat-10lap-*.json`.

### 117 — Spatial road-centre continuity on the failing seed

Hypothesis: the extractor's 0.24 maximum row-to-row centre-jump fraction may
admit spatially misplaced road regions; tightening it could reject the
texture-dependent mask bias without temporal lag or speed planning.

| Maximum centre jump | Mean speed (m/s) | Off-road |
|---:|---:|---:|
| 0.24 incumbent | 2.339 | 9 |
| 0.15 | 2.418 | 4 |
| 0.10 | **2.469** | **1** |
| 0.08 | 2.452 | 2 |
| 0.09 | 2.373 | 7 |

The 0.10 setting substantially improved both metrics but missed the strict
zero-exit gate. Nearby 0.08 and 0.09 settings regressed, showing a
discontinuous phase-sensitive boundary rather than a robust optimum.

Status: **rejected; centre-jump fraction restored to 0.24 and original seed
restored**. A future solution should use multi-frame projected-boundary
consistency with explicit hysteresis, not a single hard spatial threshold.

Evidence: `build/benchmarks/race-opt-seed-b-centre-jump-*.json`.

### 118 — Ten-lap live asynchronous deployment boundary

The one-lap 2.45 m/s certificate was extended to ten live-asynchronous laps.
At a 2.45 request, Waveshare and technical passed but tight hairpin recorded
two exits. Hairpin bracketing then found:

- 2.300 and 2.375 m/s requests: zero exits over ten laps;
- 2.39375 and 2.4125 m/s requests: two and three exits, respectively.

A formal all-four-track certificate used ten laps per track at 2.375 and
2.393 m/s. **2.375 passed all 40 laps with zero exits; 2.393 failed with one
hairpin exit.** The exact registry entry is now **2.375 m/s simulated / 1.900
m/s provisional deployment**, replacing the less robust 2.45 / 1.96 one-lap
entry. Under hot asynchronous throughput, observed target-track means at the
2.375 request were Waveshare 2.161, technical 1.951, and hairpin 2.151 m/s;
the governor correctly slowed as inference throughput fell.

Status: **accepted as the live deployment request limit**. This does not
change the deterministic 200 Hz-camera controller comparison or its 2.5 m/s
simulation request.

Evidence: `build/benchmarks/current-robust-live-speed-certificate-40p625-2026-08-03.json`
and `build/benchmarks/race-opt-final-live-async-speed-*.json`.

## Final accepted outcome

Accepted stack: SegFormer-B0 Core ML FP16 384 px, deterministic actual-model
scheduling from the 200 Hz simulated camera, pure pursuit, temporal filter
off, minimum-time racing line with nine lateral candidates/full-row road
extraction, curvature speed planner off, 30 rad/s steering-command limit, and
40.625 m/s² simulation-only governor acceleration. The real JetRacer Pro
profile still caps acceleration to 0.8 m/s².

| Track | 10-lap mean (m/s) | 30-lap mean (m/s) | 30-lap mean deviation (m) | 30-lap off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 2.483584 | 2.494503 | 0.03969 | 0 |
| Technical chicane | 2.491492 | 2.497139 | 0.03898 | 0 |
| Tight hairpin | 2.483469 | 2.494469 | 0.03978 | 0 |

The accepted 30-lap macro mean is **2.495370 m/s**, with zero exits over 90
target-track laps. This is **+127.35%** over the 1.0976 m/s controlled campaign
baseline and is within **0.185%** of the configured 2.5 m/s vehicle ceiling.
Relative to the deterministic 8/8 starting point late in the campaign, the
ten-lap macro improved from 2.474793 to **2.486182 m/s** (+0.46%).

The current robust asynchronous high-speed certificate is **2.375 m/s
simulated / 1.900 m/s provisional deployment** (ten laps per track, including
the open oval). Complete selectable-matrix coverage is
30/30, with all cases honestly marked uncertified under the separate
non-monotonic 0.5–2.5 m/s full policy. The alternate-seed study shows that
texture-sensitive path geometry—not nominal-seed speed—is now the principal
remaining simulator limitation.

### 119 — Restore realistic deployment acceleration and braking

The simulation governor's 40.625 m/s² acceleration was an optimization-only
value that did not represent the JetRacer Pro. It was restored to the
provisional hardware envelope: **0.8 m/s² acceleration and 3.0 m/s²
deceleration**. No controller or steering parameter changed.

The live asynchronous Core ML FP16 384 px lane benchmark requested 2.375 m/s
for ten laps on each target track, using the 200 Hz simulated camera with
object detection disabled:

| Track | Mean speed (m/s) | Mean deviation (m) | Inference FPS | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 2.119888 | 0.03160 | 118.69 | 0 |
| Technical chicane | 1.781555 | 0.01926 | 95.49 | 0 |
| Tight hairpin | 1.657747 | 0.03072 | 89.43 | 0 |

All **30/30 laps passed with zero exits**. The macro mean was **1.853063
m/s**. The earlier 40.625 m/s² live run averaged 2.087974 m/s, but the two
live runs had different measured inference rates, so the 11.25% macro change
is not an isolated acceleration effect. A fixed-throughput paired run is
needed for that attribution; this run is the more realistic deployment
measurement.

Status: **accepted as the deployment benchmark actuator envelope**.

Evidence: `build/benchmarks/realistic-actuators/*-10lap.json`.

### 120 — Fixed-throughput actuator-limit attribution

To isolate acceleration from live inference-rate and thermal variation, the
realistic 0.8 m/s² limit and the old 40.625 m/s² counterfactual were run with
identical deterministic actual-model scheduling: SegFormer-B0 Core ML FP16
384 px, 128 FPS governor telemetry, 12 ms perception latency, 2.375 m/s
request, and ten laps per track. Deceleration remained 3.0 m/s².

| Track | 0.8 m/s² mean | 40.625 m/s² mean | Counterfactual gain | Deviation: 0.8 / 40.625 m | Off-road |
|---|---:|---:|---:|---:|---:|
| Waveshare 3×2 | 2.226583 | 2.360354 | +6.01% | 0.03586 / 0.03866 | 0 / 0 |
| Technical chicane | 2.296077 | 2.367387 | +3.11% | 0.03458 / 0.03546 | 0 / 0 |
| Tight hairpin | 2.225248 | 2.360211 | +6.07% | 0.03752 / 0.03818 | 0 / 0 |

The realistic macro mean was **2.249303 m/s**, versus **2.362651 m/s** for
the idealized actuator: an isolated **0.113348 m/s (5.04%)** advantage. Both
completed all 30 laps without exits, but the idealized actuator increased
mean centre deviation on every track. The speed gain is therefore an
unrealistic launch/speed-recovery benefit, not a lane-control improvement.

Status: **realistic 0.8 m/s² acceleration retained**; 40.625 m/s² remains a
counterfactual result only. The active deceleration limit remains 3.0 m/s².

Evidence: `build/benchmarks/actuator-limit-paired/*.json`.

### 121 — Single-cylinder avoidance controller

A deterministic matte-magenta cylinder was added to every benchmark track.
Its visual radius, collision radius, height, placement seed, and placement
range are configuration values. The reference geometry is **30 mm radius ×
120 mm height**.

The no-avoidance baseline collided once per lap on all four tracks (20/20
encounters). A fixed 84 mm offset avoided the cylinder on Waveshare and the
open oval, but collided on all five technical-chicane and all five tight-
hairpin encounters. The accepted controller uses measured ground position,
chooses the feasible side with the largest remaining road corridor, follows a
spatial detour, adds up to 20 mm clearance on high-curvature road, and releases
the detour from forward ground distance rather than an image-loss timer.

| Track | Mean speed (m/s) | Mean deviation (m) | Collisions | Off-road |
|---|---:|---:|---:|---:|
| Waveshare 3×2 | 0.717 | 0.061 | 0 | 0 |
| Open oval | 1.502 | 0.037 | 0 | 0 |
| Technical chicane | 1.028 | 0.046 | 0 | 0 |
| Tight hairpin | 0.512 | 0.035 | 0 | 0 |

Status: **accepted in oracle-perception simulation**; all 20 laps passed with
zero collisions and zero off-road events. Real-camera color/ground-projection
calibration remains a hardware-arrival task.

Evidence: `build/benchmarks/cylinder/*-5lap-all-tracks*.json`.

### 122 — Cylinder placement and perception robustness

The original 20-lap result repeatedly exercised one deterministic placement
per track. Two frozen, reproducible suites now test spatial generalization and
joint parameter uncertainty at the 200 Hz simulation rate.

The nominal placement suite exhaustively covers the configured 7 × 7 grid on
each of four tracks: seven track fractions from 0.2 through 0.8 and seven
lateral offsets from -75 through +75 mm. Its 196 one-lap cases use the 30 mm
radius × 120 mm reference cylinder, recommended track speed, and perfect
oracle obstacle measurements.

| Track | Safe cases | Safe fraction |
|---|---:|---:|
| Waveshare 3×2 | 21/49 | 42.9% |
| Open oval | 45/49 | 91.8% |
| Technical chicane | 42/49 | 85.7% |
| Tight hairpin | 15/49 | 30.6% |
| **Total** | **123/196** | **62.8%** |

Of the 73 unsafe placements, 9 had collision only, 19 had off-road only, and
45 had both. Every centre-offset case failed on Waveshare, the technical
chicane, and the tight hairpin. Failure also varies sharply with track
fraction, showing that local curvature and detour geometry dominate the
current controller's outcome.

The second suite contains 100 deterministic joint Monte Carlo cases spanning
20–40 mm visual radius, up to 5 mm extra collision radius, 60–180 mm height,
0.75–1.20× speed, 0–80 ms latency, up to 20% periodic dropout, -10% to +15%
range bias, ±12 mm lateral bias, and random placement. It passed **58/100**:
10/25 Waveshare, 24/25 open oval, 19/25 technical chicane, and 5/25 tight
hairpin. Absolute lateral placement was the strongest sampled correlate: the
most central offset quartile passed 28%, versus 84% for the outermost
quartile. Perception latency and dropout were not dominant within these
confounded joint samples.

Status: **the prior single-placement acceptance does not generalize**. The
robustness machinery is accepted, but the clearance-aware controller is not
safe for an instruction that permits placing the cylinder anywhere on the
road. Height-dependent image detectability remains untested because these
runs use oracle color-object measurements with injected faults.

Evidence:
`build/benchmarks/cylinder/cylinder-placement-grid-196.json` and
`build/benchmarks/cylinder/cylinder-robustness-100.json`.

### 123 — Waveshare centreline cylinder-avoidance optimization

The frozen Waveshare 7 × 7 placement grid was used as the acceptance test.
A case is safe only when its one-lap run has zero collisions and zero
off-road incidents. All changes retained the centreline planner and the
realistic vehicle model.

| Controller revision | Safe cases | Safe fraction | Grid mean speed (m/s) |
|---|---:|---:|---:|
| Original robustness baseline | 21/49 | 42.9% | 0.7401 |
| Footprint-aware centred pass | 23/49 | 46.9% | 0.7397 |
| Distance-based 0.18 m egress | 26/49 | 53.1% | 0.7403 |
| Curvature-gated quintic ingress | 29/49 | 59.2% | 0.7393 |
| Curvature-only 0.8 speed scale | 31/49 | 63.3% | 0.6949 |
| Earlier steering at 2.4 m | **32/49** | **65.3%** | **0.6912** |

The final change separates the steering and slowdown thresholds. The car
starts constructing the detour at 2.4 m, but the 0.8 high-curvature speed
scale still begins at 2.0 m. It recovered placement 007 without losing any
of the 31 previously safe placements. Relative to the 31-case controller,
the grid-wide mean fell 0.52% because the curvature speed planner observes
the detour sooner. The required obstacle-free five-lap regression was
unchanged at exactly **0.8966139836 m/s**, with zero collisions and zero
off-road incidents.

Rejected experiments were a cubic ingress profile (29/49), shorter 0.12 m
and 0.09 m egress distances (no targeted gains), and a 2.3 m steering trigger
(31/49 after regressing placement 039). A 2.5 m trigger also achieved 32/49
but had a lower 0.6908 m/s grid mean than the accepted 2.4 m setting.

Status: **accepted at 32/49 for continued development, not deployment-safe**.
The remaining 17 unsafe placements contain 9 collision cases and 15
off-road cases, with overlap.

Evidence:
`build/benchmarks/cylinder/waveshare-placement-grid-early-steering-2p4-49.json`
and
`build/benchmarks/cylinder/waveshare-obstacle-free-after-early-steering-2p4.json`.

### 124 — Footprint constraint and slowdown feasibility probes

A spatial footprint transition constraint was added as an experimental,
configurable option. It derives the minimum quintic lane-change distance from
the configured road width, vehicle length and width, road margin, requested
lateral offset, and the rotated rectangular chassis projection. Applied to
all 17 failures from experiment 123, it recovered **0/17**. It remains covered
by unit tests but is disabled by default because it has no demonstrated
benchmark benefit.

Two lower-speed counterfactuals then reran all 17 failures with unchanged
geometry. A 0.8 m/s cruise recovered **0/17**; a 0.6 m/s cruise also recovered
**0/17**. Lower speed sometimes converted an off-road event into a collision
but never produced a safe case. Slowing is therefore not the primary remedy
for this failure set.

The configured Stanley controller was also injected correctly as an
alternative to pure pursuit. It largely kept the car inside the road but all
17 cases still collided with the cylinder. This isolates the bottleneck to
the detour reference: heading/cross-track feedback can choose road retention
or obstacle clearance, but the current shifted-centreline path does not
provide both simultaneously.

Status: **rejected as production changes**. The accepted default remains the
32/49 controller from experiment 123. The next planner must optimize a full
local path against both road-boundary and obstacle swept-footprint constraints
instead of modifying one transition-distance scalar.

### 125 — Camera-local swept-footprint trajectory prototype

A camera-local candidate optimizer was implemented behind a disabled feature
flag. It propagates the detected cylinder's road-relative forward/lateral
position and radius into steering, evaluates multiple offsets and quintic
transition lengths, and rejects candidates when the rear-axle-referenced
JetRacer rectangle crosses a detected road boundary or intersects the
cylinder. Unlike experiment 124, the body envelope correctly uses the full
255 mm forward extent from the rear axle. Unit tests verify that it finds a
collision-free route in a synthetic straight 550 mm road corridor.

The optimizer recovered **0/17** previously unsafe Waveshare placements. In a
representative failure it attempted 879 planning frames and evaluated 49,224
candidates: 49,015 violated the road envelope, 191 hit the obstacle, and only
18 were feasible. A changed path was emitted on three frames, too briefly to
alter the outcome. Allowing the optimizer to switch pass side also recovered
**0/5** representative failures.

The unoptimized prototype increased a simulated lap's wall time from roughly
5 seconds to 6–38 seconds. This is not acceptable for a 200 Hz controller even
if safety had improved.

Status: **prototype retained but disabled; production result remains 32/49**.
The result shows that independent per-frame path selection is insufficient.
A useful next version needs a committed receding-horizon trajectory state,
vehicle-motion propagation between camera frames, and an optimized/vectorized
solver rather than a stateless candidate sweep.

### 126 — Fail-safe committed hybrid local planner

Planner infeasibility now reaches longitudinal control. A configurable
reaction-plus-braking envelope stops before the cylinder, and the stop remains
latched until the same obstacle is cleared. This established a **49/49 safe
floor with zero collisions and zero off-road events**, but initially stopped
in all 49 placements.

Persisting the selected swept-footprint offset through the complete pass
recovered 5 full laps. A hybrid lattice then prefers road-aware candidates and
falls back to obstacle-only candidates; the benchmark's exact relaxed-boundary
test remains the final road-safety authority. Increasing the sampled extra
lateral offset from 0.06 m to 0.12 m raised the accepted result to **24/49 full
laps and 25 controlled stops**, still with zero collisions and zero off-road
events. Completed laps averaged **0.7953 m/s** and the minimum exact obstacle
clearance was **27.8 mm**.

Rejected variants:

- bounded post-obstacle validation horizon: representative completions 5 to 4;
- local bump trajectories: representative completions 5 to 3;
- minimum-heading-first side cost: collision in placement 004;
- Stanley lateral control: collisions in the first two constrained cases;
- 0.32 minimum speed scale: collision in placement 004;
- 0.32 speed plus 15 mm reserve: 23/49, below the 24/49 incumbent;
- 0.13-0.15 m extra lateral authority: off-road regressions.
- 2.0-2.4 m dense long-ingress lattice: no targeted gains and regressions in
  established placements 005 and 040.
- 13-17 lateral candidates inside the same 0.12 m bound: recovered placement
  012 but regressed established placement 040; the 9-candidate lattice remains
  the accepted speed/safety trade-off.

Status: **accepted for continued simulation development at 24/49 completion,
49/49 safe outcomes**. It is not ready for real deployment because 25 cases
still stop and 27.8 mm is not yet a sufficient physical tracking/cable margin.

Evidence:
`build/benchmarks/cylinder/waveshare-placement-grid-hybrid-wide-49.json`.

### 127 — Gated bicycle-rollout fallback

A short-horizon rear-axle bicycle rollout was added as an optional second
trajectory model. It projects sampled detour paths through the configured
wheelbase and steering limit before applying the existing rectangular
swept-footprint collision and relaxed-road checks.

The rollout model alone regressed the first 15 placements from four to three
completions. A geometric-first fallback recovered placements 006, 012, 015,
and 027, but an unreserved version collided in placement 031 and a 20 mm
tracking reserve still collided in placement 039. Committing to the rollout
model after selection removed mode-switching collisions in placements 031 and
039, but exposed the same failure for the centred obstacle in placement 004.

The collision set was concentrated at near-centre obstacle offsets, whereas
the stable gains were at absolute offsets of 50-75 mm. The accepted planner
therefore enables rollout fallback only beyond a configurable 40 mm absolute
road-relative obstacle offset, commits to that model for the encounter, and
uses a configurable 30 mm rollout tracking reserve. Near-centre cases retain
the proven geometric-plan-or-stop behavior.

| Planner | Full laps | Controlled stops | Collision/off-road | Completed mean speed |
|---|---:|---:|---:|---:|
| Geometric hybrid incumbent | 24/49 | 25 | 0 / 0 | 0.7953 m/s |
| Gated committed rollout | **27/49** | **22** | **0 / 0** | 0.7932 m/s |

The added completions are placements 006, 015, and 027. Grid-wide mean speed
rose from 0.6425 to 0.6560 m/s, minimum exact obstacle clearance increased
from 27.8 to 33.5 mm, and total matrix wall time increased 5.3% (373.3 to
393.3 seconds). The obstacle-free five-lap Waveshare regression passed with
zero incidents at 0.89664 m/s, effectively unchanged from 0.89661 m/s.

Status: **accepted for continued simulation development at 27/49 completion
and 49/49 safe outcomes**. The rollout remains gated and is not yet suitable
for unrestricted real-hardware use.

Evidence:
`build/benchmarks/cylinder/waveshare-placement-grid-hybrid-gated-rollout-margin30-49.json`
and
`build/benchmarks/cylinder/waveshare-obstacle-free-hybrid-gated-rollout-regression-5lap.json`.
