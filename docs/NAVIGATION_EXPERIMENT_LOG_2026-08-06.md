# Navigation experiment log — 2026-08-06

Scope: centre-line driving and one-to-three known cylinders, with no stop signs.
All reported safety results require completion, zero collisions, and zero
off-road events under the relaxed real-track boundary policy.

## Accepted changes

| Change | Evidence | Effect |
|---|---|---|
| Structured segmentation fault injection | Seeded band jitter, row dropout, occlusion rectangles, false-road rectangles, and periodic full-mask dropout | Provides repeatable, inexpensive imperfect-perception tests without running a trained network |
| Temporal path filtering for noisy masks | Four-track moderate-noise screen, one lap | Mean speed rose from 1.003 to 1.066 m/s and deviation fell from 27.3 to 18.1 mm for adaptive Pure Pursuit; zero off-road events in both cases |
| Temporal filtering with obstacle avoidance | Same 30 randomized layouts and combined moderate mask/detector faults, paired off versus on | Safe completion rose from 22/30 to 26/30 and mean speed from 0.562 to 0.596 m/s; zero collisions in both |
| Adaptive Pure Pursuit | Four tracks, five clean laps | Mean deviation fell from 18.24 to 14.97 mm versus fixed Pure Pursuit (18.0%) at the same 1.084 m/s mean speed; zero off-road events |
| Adaptive Pure Pursuit under noise | Four tracks, five noisy laps with temporal filtering | Mean deviation fell from 19.72 to 17.80 mm versus fixed Pure Pursuit (9.7%) at the same 1.083 m/s; zero off-road events |
| Explicit avoidance handover state | Target hysteresis, pass-side continuity, approach/hold/egress phases, 50 ms release confirmation, and 80 ms steering blend | Prevents target and controller chatter between adjacent objects |
| Multi-object common corridor | One, two, and three differently coloured cylinders | Makes the avoidance offset consistent across a visible obstacle chain |
| Deterministic randomized multi-object layouts | Ten configured layouts per object count, with seeded placement, lateral bounds, and minimum spacing | Exposed a collision hidden by the three hand-picked layouts and provides a reproducible 30-case Waveshare robustness gate |
| Automation-safe multi-object benchmark | Reports include a top-level pass state and return non-zero for unsafe cases unless `--exploratory` is selected; headless CI runs the combined-noise fixed matrix on every configured track | Makes the matrix a regression gate rather than report-only tooling and automatically covers future configured tracks |
| DWA local planner | Compared with hybrid lattice, hybrid bicycle rollout, and obstacle-only lattice | Only DWA was promising in the initial planner screen; the lattice variants either left the road or stopped before completing |
| Transient infeasibility debounce | 25 ms confirmation before latching emergency braking | Removed false permanent stops caused by a single 5 ms infeasible plan while retaining stop-before-obstacle behaviour for persistent infeasibility |
| DWA negative-cache fix | Replan an empty plan after normal pose-change threshold | Prevented one transient planning miss from being cached for the entire encounter |
| Two-arc DWA fallback | Opposite-steering second arc is considered only when all single arcs fail | Adds a bounded recovery option without changing successful single-arc plans |
| Density-aware handover speed scale | Persistent 0.85 multiplier while an obstacle chain is active | Three-cylinder clean/noisy cases remain complete and collision-free; 0.90 caused an off-road event and 0.80 was slower |
| Size-aware obstacle margin | 7.5 mm for small cylinders; 30 mm when physical radius is at least 45 mm | Raised randomized clean completion from 29/30 to 30/30 for a 0.08% aggregate speed cost; pedestrian clearance remains 52.9 mm and physical radius stays separate from DWA tracking-error inflation |
| High-curvature DWA margin selected by the avoidance profile | Non-linear ingress profile plus road width at least 0.45 m selects a 15 mm minimum margin for the full encounter | Removed the remaining noisy tight-hairpin collision without making the constrained Waveshare three-object corridor infeasible; all 12 clean and all 12 noisy multi-object cases pass |
| Zero-cost normal/avoidance composition | Four tracks, three obstacle-free laps, bare adaptive controller versus handover wrapper | Every speed, deviation, and steering metric was numerically identical; the wrapper is inactive until avoidance begins |
| Adaptive handover + DWA defaults | Full 11-case oracle acceptance suite | All lane, stop, pedestrian, and cylinder cases pass; fixes the legacy hybrid roll-out collision on the tight hairpin |

## Final benchmark gates

| Gate | Result |
|---|---|
| Waveshare single-cylinder placement grid with selected controller | 49/49 safe, 0.701 m/s mean, 46.9 mm minimum clearance |
| Waveshare randomized one/two/three-cylinder layouts, clean | 30/30 safe, zero collisions/off-road events, 0.548 m/s mean |
| Waveshare randomized layouts, combined moderate mask and obstacle noise | 26/30 safe completions, zero collisions; three fail-safe stops and one off-road completion remain |
| One/two/three cylinders across all four tracks, clean | 12/12 safe; Waveshare 0.754 / 0.437 / 0.368 m/s |
| Waveshare one/two/three cylinders, combined moderate mask and obstacle noise | 3/3 safe; 0.808 / 0.507 / 0.437 m/s |
| One/two/three cylinders across all four tracks, combined moderate mask and obstacle noise | 12/12 safe; zero collisions and zero off-road events |
| Adaptive driving, adversarial segmentation noise | Four tracks × five laps, all complete, zero off-road events |
| Full oracle acceptance | 11/11 pass, including pedestrian clearance and tight-hairpin cylinder avoidance |
| Waveshare fixed one/two/three-cylinder endurance, clean | 3/3 complete over five laps each (15 total), zero collisions/off-road events |
| Waveshare fixed endurance, combined moderate noise | One-cylinder completes five laps; two cylinders stop safely at 2.21 laps and three cylinders at 3.16 laps, zero collisions/off-road events |
| Waveshare joint cylinder Monte Carlo | 25/25 safe with randomized radius, height, collision envelope, speed, latency up to 79 ms, dropout, range bias, and lateral bias; 0.729 m/s mean and 32.2 mm minimum clearance |

The noisy multi-object speeds can be higher than clean speeds because the fixed,
seeded range bias changes when the handover slowdown is entered; this is not
interpreted as a performance gain from noise.

## Recommended planner redesign

The next local planner should optimize steering and speed in the same receding
horizon. Each candidate must use the measured steering-actuator state, the
configured acceleration/deceleration limits, the rectangular swept footprint,
the relaxed road-boundary rule, and an obstacle uncertainty interval derived
from detector age and calibrated range error. Progress, centreline recovery,
clearance, and control variation belong in one objective; collision and the
relaxed road boundary remain hard constraints. This directly addresses the
fixed-speed DWA mismatch exposed by the noisy randomized layouts.

Acceptance should require the 49-case placement grid, 30 clean randomized
layouts, 30 combined-noise randomized layouts, the four-track fixed
one/two/three-object matrix, and five-lap Waveshare endurance, with zero
collisions. The current DWA remains the deployment default until a replacement
improves these gates without reducing obstacle-free speed.

## Rejected or bounded experiments

| Experiment | Result | Conclusion |
|---|---|---|
| LQR normal tracker | Failed to complete the tight-hairpin screen; 105 mm macro deviation | The current local quadratic reference fit is unsuitable for sharp visible-path topology; retain as experimental |
| LQR avoidance tracker | Safe for one and two objects, but did not complete the three-object case; 0.423 m/s mean | Pure Pursuit/DWA handover is safer and faster |
| Stanley lane tracker | Completed but had 32.8 mm macro deviation in the controller screen | Inferior to tuned adaptive Pure Pursuit |
| DWA as an always-on lane tracker | Completed but had 75.0 mm macro deviation | Use DWA only for local obstacle manoeuvres |
| Hybrid lattice / bicycle rollout / obstacle-only lattice | 0/3 safe completions in the initial multi-object screen | Reject for the current short camera horizon and actuator model |
| DWA horizon 1.2 s | Introduced collision/off-road failures | The longer open-loop projection is too inaccurate; retain 0.8 s |
| Handover speed scale 0.90 | Three-object noisy case left the road | Retain 0.85 |
| Handover speed scale 0.80 | Safe but slower | Retain 0.85 |
| Timed rather than persistent slowdown | Clean cases could pass, but noisy three-object case left the road | Keep slowdown active through the complete obstacle chain |
| Uniform 30 mm margin for every obstacle | Three-cylinder case completed but left the road | Apply the large margin only to physically larger objects |
| Speed-only high-curvature mitigation | Reducing the hairpin avoidance scale from 0.80 to 0.65 reduced speed but did not prevent collision | Collision geometry, not speed alone, was the limiting factor |
| Duplicate curvature estimate inside DWA | Fixed the hairpin in isolation but falsely triggered on Waveshare/oval perception geometry | Reuse the avoidance controller's existing curvature classification instead of adding a noisy second estimator |
| Global 10% range-overestimate correction | Recovered the three early-curve stops in isolation but introduced collision/off-road failures elsewhere under combined noise | Leave the configurable correction at zero until real range calibration provides a measured bias |
| DWA road-boundary hard constraint / soft cost | Hard rejection reduced noisy completion to 20/30; soft cost introduced collisions | Retain the relaxed policy and treat road-aware DWA/MPC as a separate planner redesign rather than a scoring patch |
| Lower DWA planning-speed floor | 0.20 m/s recovered some stops but introduced a collision; 0.40 m/s added an off-road event without recovery | Retain the validated 0.60 m/s planning floor and fail-safe stop behavior |
| Dense-chain speed heuristic | Fixed one noisy off-road case but slowed clean cases and eventually caused a clean collision because commanded speed fell below DWA's planning floor | Rejected; speed and trajectory prediction must remain coupled |
| Detector-age ego-motion compensation | Recovered the three original noisy fail-safe stops, but moved DWA infeasibility to four other randomized layouts; overall completion fell from 26/30 to 25/30 with the same one off-road event | Reject coordinate-only compensation; the remaining discontinuity needs an uncertainty-aware planner with speed included in its roll-out state |
| Expanded two-arc DWA fallback lattice | Recovered the one-object noisy stop, but produced seven failures in the first 24 randomized cases versus four for the baseline over the same prefix | Rejected early; longer action sequences amplify model/tracking mismatch and do not solve common-corridor target discontinuities |
| DWA angular sampling 11 to 21 arcs | All four known noisy failures reproduced unchanged | Rejected; failures are not caused by coarse yaw-rate discretization |
| Small-obstacle margin 7.5 to 6.25 mm | Clean randomized layouts remained 30/30 and two targeted noisy stops recovered, but the full noisy matrix fell from 26/30 to 24/30 as three-object plans became infeasible | Rejected; DWA plan selection is non-monotonic near clearance boundaries, so targeted placement wins are insufficient |
| Longer egress / shorter post-pass hold | Neither a 0.30 m egress nor an 80 mm geometry-based hold changed the sole noisy off-road case | The departure occurs while the obstacle plan remains active, not during the return-to-lane handover |
| Waveshare high-curvature slowdown threshold 1.5 to 1.2/m | Reduced the affected case from 0.670 to 0.604 m/s but reproduced the same road departure | Rejected; speed alone does not correct the active local path geometry |
| Centred-obstacle side-selection band 5 to 30 mm | All four known noisy failures reproduced unchanged | Rejected; pass-side selection is not the limiting factor in these cases |
| DWA horizon 0.8 to 0.6 s | Fixed the noisy road departure, but caused a collision and off-road event in clean randomized three-object layout 5 | Rejected; the extra 0.2 s of collision look-ahead is necessary even though it is conservative under noise |
| DWA horizon 0.7 s | Preserved the clean 0.6 s failure case but did not fix the noisy road departure | Rejected; retain the safer 0.8 s horizon |
| Actual Cityscapes SegFormer certification | Failed minimum 0.5 m/s on Waveshare: 6 off-road events without filtering, 10 with temporal filtering, despite about 130 completed masks/s | This is systematic sim-domain semantic mismatch rather than an FPS or flicker problem; keep deployment uncertified and fail-closed |

## Output files

Key reports are under `build/benchmarks/`, including:

- `overnight-pp-vs-adaptive-clean-4tracks-5laps.json`
- `overnight-pp-vs-adaptive-moderate-noise-temporal-4tracks-5laps.json`
- `overnight-adaptive-adversarial-noise-4tracks-5laps.json`
- `overnight-adaptive-vs-handover-clean-4tracks-3laps.json`
- `cylinder/overnight-waveshare-placement-grid-adaptive-dwa-final-49.json`
- `overnight-waveshare-random-multi-obstacle-final-clean-30.json`
- `overnight-waveshare-random-multi-obstacle-final-moderate-noise-30.json`
- `overnight-waveshare-random-multi-obstacle-noisy-temporal-off-30.json`
- `overnight-multi-obstacle-final-all-tracks-clean.json`
- `overnight-multi-obstacle-final-all-tracks-moderate-noise.json`
- `overnight-waveshare-multi-obstacle-endurance-clean-5laps.json`
- `overnight-waveshare-multi-obstacle-endurance-moderate-noise-5laps.json`
- `overnight-waveshare-joint-cylinder-monte-carlo-final-25.json`
- `overnight-ci-multi-obstacle-acceptance.json`
- `overnight-oracle-full-acceptance-final.json`

## Integration validation

- All 173 Python tests and all 47 non-evidence CTests pass; the native build
  succeeds.
- Final replays of both deterministic 30-layout Waveshare matrices were
  numerically identical to their accepted reports: clean remained 30/30 and
  combined moderate perception noise remained 26/30, with zero collisions.
- Controller construction is shared by the control, multi-object, speed,
  stop-sign, and camera-mount runners. It supports adaptive Pure Pursuit, LQR,
  Stanley, DWA, and recursive handover configurations; the camera CTest smoke
  passes.
- The legacy NumPy closed-loop smoke is capped at its verified 1.2 m/s and
  passes. Its previous 1.8 m/s setting consistently lost the road.
- Two fresh headless Core ML runs sustained 200.0 source FPS and 56.8--59.5
  inference FPS with zero failed or discarded frames. The final run passed the
  55 ms perception-age gate at 52.9 ms but still failed the unchanged 25 ms
  p99 inference-latency gate at 28.8 ms. It was not promoted.
- The exhaustive speed-certification catalog remains stale/missing for 144
  combinations. It and passing performance evidence must be regenerated
  separately before the entire CTest suite can be green.
