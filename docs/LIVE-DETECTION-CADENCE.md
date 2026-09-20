# Configurable live detection and reference benchmarks — 2026-09-20

The demo now lets the user choose **Detect person every N frames**, from 1 to
30, before starting a live session. The initial UI value is 5. ViTPose and GEM
still run on every processed frame; only YOLOX is skipped, with the last crop
reused. This suits a person remaining in the same area, not walking around.
It is not a tracker. Offline processing is unchanged.

The native worker and HTTP API retain interval 1 when no interval is supplied.
This preserves existing callers and the upstream detection cadence. An explicit
interval 1 produces all 215 replay poses byte-identical to the accepted baseline.

## Native measurements

RTX 5070 Ti, strict F32, CPU affinity 0–7, eight compilation jobs maximum.
The public 72-frame football clip is replayed three times: 216 processed frames,
215 output poses and 187 full-window frames used for steady-state timings.

| Detection interval | Detector calls / 216 frames | Mean worker time | Native throughput | p95 worker time |
| --- | ---: | ---: | ---: | ---: |
| Every frame | 216 | 113.920 ms | 8.78 fps | 114.975 ms |
| Every 5 | 44 | 69.908 ms | 14.30 fps | 114.081 ms |
| Every 10 | 23 | 64.697 ms | 15.46 fps | 113.951 ms |

Every 5 frames gives about **1.63× throughput**; every 10 gives **1.76×**.
Refresh frames still pay the detector cost, so p95 latency barely changes.
These measurements exclude camera, browser, image encoding and network costs.
The initial live browser rate cap is now 20 fps, so a 10 fps cap does not mask
the benefit; offline mode starts at its existing 10 fps sample rate.

Misses trigger another detection on the next frame, explaining deviations from
an exact one-in-N count. Explicit reset, changed image dimensions and a pause
over two seconds clear the cadence state and trigger a fresh detection.

## Quality scope

A stationary control repeats the same real image for 216 frames. Intervals 1
and 5 produce all 215 complete poses byte-identically, with mean worker times
113.642 and 69.708 ms. This establishes unchanged results for an unchanged
crop/image, not quality on a stationary person moving their limbs.

The moving football clip is a poor fit for crop reuse and shows **large
deviations**. Compared with detection every frame, mean 2D keypoint displacement
is 19.67 px at interval 5 and 51.15 px at interval 10. Mean pelvis-relative joint
displacement is 229 mm and 314 mm respectively. These statistics cover all 77
joints over 215 outputs, including clip wrap boundaries and low-confidence
keypoints; they are differences from the baseline, not errors against ground
truth. No representative physical stationary-teleoperation clip was supplied.
Use interval 1 when moving around or when matching upstream behaviour matters.

## Independent implementation benchmark

The existing `gemx-reference:e2e` container runs ONNX Runtime 1.23.2 with CUDA.
The same existing ONNX model artifacts are used. No alternative backend is
integrated into the application. Each result is 100 timed calls after 20
warm-ups, in a fresh process, with eight-core affinity and bounded CPU threads.

| Model | ORT CUDA, TF32 disabled | ORT CUDA, TF32 enabled | Native reference point |
| --- | ---: | ---: | ---: |
| ViTPose, crop + flip | 40.866 ms | 24.564 ms | 54.508 ms normalized inference |
| YOLOX | 29.829 ms | 17.395 ms | 55.463 ms live detector stage |

ORT timings include `session.run` host input/output transfers and exclude
preprocessing. The native detector stage includes preprocessing and native
postprocessing; the ONNX detector includes its exported postprocessing. These
are reference points, not a fully matched end-to-end speedup comparison. Models
load outside the timed interval, and the script refuses whole-session CPU
fallback. Shape/control operations may still execute on CPU.

Strict CUDA ViTPose matches the pinned fixture closely: maximum heatmap error
7.30e-7, mean 1.03e-8, and zero changed raw heatmap peaks. Enabling TF32 gives
maximum error 2.20e-4, mean 3.42e-6, and **two changed raw heatmap peaks out of
154**. This single fixture does not establish motion quality. Detector quality
and end-to-end motion under ORT/TF32 were not audited in this benchmark.

The reference exposes substantial headroom in both native models. It does not
demonstrate that a particular Vulkan change will recover the same gains.

## Controls, validation and reproduction

- Worker: append `[DETECT_INTERVAL]` after the directory argument to
  `gemx-pipeline --live-worker`; omitted means 1.
- HTTP: `POST /api/live?detect_interval=N`; omitted means 1. Invalid values
  return 400 before acquiring the GPU.
- UI: choose 1–30 before starting; stop the session to change the setting.
- Replay: `scripts/profile_live_worker.py --detect-interval N --save-poses`
  records detector calls, stage timings and outputs for comparisons.
- Reference only: `scripts/benchmark_onnx_reference.py --help` documents the
  standalone ORT benchmark. Models and CUDA dependencies remain external.

Five CTest contracts and the Go tests pass. Tests cover cadence, immediate retry
after missed detection, reset, interval validation and propagation into the
worker. Python syntax and whitespace checks pass.

The updated demo is deployed at https://gem-x.d.richiejp.com. Chromium smoke
testing selected interval **7**, verified the outgoing request, rendered 33
poses, restarted the session, switched to offline mode and handled denied
camera permission. No JavaScript exceptions occurred. This smoke test used a
procedurally generated gradient video; there were no detected people, so it
exercises the miss/retry path rather than establishing speed or pose quality.
Only one frame request was in flight at a time.

Raw artifacts are in `generated/live-cadence-20260920/`; the committed
[measurement summary](live-cadence-2026-09-20.json) contains timings, quality
differences, hashes and browser results.
