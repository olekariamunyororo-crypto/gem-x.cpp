# Live frame pipelining — 2026-09-21

The demo now prepares/uploads one following camera frame while the current
frame runs inference. On the local simulated-camera test this raises throughput
from **13.28 to 15.35 fps (+15.6%)**, with roughly **5.5 ms additional frame age**.
The shared camera viewport displays the exact captured image with its pose;
stopping inference returns it to the raw webcam.

## Matched measurements

RTX 5070 Ti, strict F32, eight CPU cores, default ViTPose optimizations,
detection every five processed frames and a 20 fps cap. Chromium uses the
pinned football clip as a simulated webcam. Browser timings exclude the first
30 frames. The same external Nsight Systems collector samples system-wide
GPU counters at 10 kHz for eight seconds after warm-up, with no competing
inference. These are localhost measurements, not physical camera or WAN latency.
The browser runs sample different source frames as throughput changes;
fixed-input parity is checked separately below.

| Metric | Serial | Bounded lookahead |
| --- | ---: | ---: |
| Delivered frames/s | 13.28 | 15.35 |
| Mean native inference, ms | 64.88 | 64.51 |
| Mean interval outside native inference, ms | 10.57 | 0.77 |
| Encode-start to response headers, mean ms | 70.98 | 76.43 |
| Encode-start to response headers, p95 ms | 115.00 | 121.50 |
| Compute in flight, % | 81.82 | 93.45 |
| SM throughput, % | 38.15 | 43.80 |
| FMA pipe throughput, % | 25.34 | 29.10 |
| Compute warp occupancy, % | 18.93 | 21.67 |
| VRAM bandwidth throughput, % | 9.50 | 10.77 |
| PCIe throughput, % | 1.11 | 1.12 |

Encode-start to headers is a latency proxy: it excludes sensor/browser capture
latency and final body parsing/rendering. The UI's frame-age measurement starts
before drawing the captured image and ends after parsing the pose body.
Compute-in-flight means compute work is outstanding; it does not mean every
arithmetic unit is occupied. Counter averages include the remaining gaps.

The final server reports mean preparation/decode/packing of **2.27 ms**, queue
wait of **5.90 ms**, and packed-file write of **0.22 ms**. Preparation now overlaps
current inference; queue wait increases frame age but does not idle the GPU.
The 0.77 ms interval estimate uses integer native-time headers and includes
response scheduling; it is not an isolated GPU idle counter.

[Machine-readable summary](live-pipeline-2026-09-21.json). Raw browser timing
arrays, screenshots, sampling commands, SQLite counter data and API traces are
under `generated/live-pipeline-20260921/`. The matched runs are
`browser-serial-matched` and `browser-adaptive`.

## What changed

Previously the browser waited for inference and its response before capturing,
encoding and uploading the next image. The server held its worker lock during
upload/decode and rejected overlap.

The default browser now holds two image buffers and at most two requests. It
predicts completion using separate measured detection/non-detection times,
encoding and transport costs. Capture begins near that completion, retaining
four milliseconds of scheduling margin. Prediction follows immediate detection
retries after a miss. It also accounts for the previous result's completion so
queueing does not accumulate. The rate cap still limits capture.

An initial early-lookahead experiment improved throughput but added almost a
full frame of latency. It was rejected. The final late-capture scheduler retains
most of the throughput benefit with a much smaller queue. Estimates affect
scheduling only; they never change model computations or skip accepted frames.

The opt-in HTTP protocol (`pipeline=2`, now used by the demo) accepts a bounded
window of numbered frames. Upload/decode/packing runs ahead of the worker lock;
file exchange, model state and pose production remain serialized. Frame numbers
prevent HTTP delivery order from changing temporal history. Duplicates and
out-of-window requests are rejected. A failed admitted frame cancels the
session. Stop, disconnection and idle expiry release queued work. Each displayed
pose retains its own immutable source image until it is rendered. Legacy HTTP
clients without the pipeline option retain serial behavior.

`Server-Timing` exposes `prepare`, `queue`, `write` and `infer` durations.
No native kernels, precision settings, model inputs or offline inference stages
changed in this pass.

## Would overlapping GPU transfers help?

It is possible in principle with separate in-flight buffers and appropriate
Vulkan synchronization, but it is a much smaller opportunity here. A separate
resident-worker API trace, averaged over 113 warm frame intervals, reports:

| CPU API operation | Mean wall time/frame |
| --- | ---: |
| Upload, about 2.19 MB/frame | 0.106 ms |
| Download, about 2.00 MB/frame | 0.343 ms |
| Graph recording/submission | 3.155 ms |
| Synchronization | 57.967 ms |

Upload/download API wall time is not isolated DMA-engine time. Nevertheless,
the observed exposed transfer calls total only **0.45 ms/frame**, versus about
10.6 ms of previously exposed browser/HTTP delay. Synchronization mostly waits
for GPU execution; its 58 ms must not be interpreted as an idle GPU.

This change overlaps browser/HTTP preparation with GPU work; it does **not** add
asynchronous GPU transfer queues. After fixing most inter-frame gaps, remaining
work should target kernel instruction/memory efficiency and graph submission,
not assume a larger input queue will fill idle arithmetic units. The existing
[QKV analysis](VITPOSE-QKV-COUNTERS.md) and ViTPose profiles remain relevant.

## Validation and reproduction

- Go lifecycle, bounded admission, duplicate/replay rejection, out-of-order
  arrival and queued cancellation checks pass with the race detector.
- Lossless 144-frame serial/pipelined replays emit **143 byte-identical complete
  poses** at detection interval 5, and another **143 identical poses** at
  upstream interval 1. This covers rolling-window eviction. It preserves the
  previously established upstream agreement; it does not establish new bit-exact
  upstream parity.
- Chromium validates live inference, two-request bound, shared viewport,
  stop-to-webcam, restart, offline mode switch and permission-error recovery,
  with no JavaScript exceptions.

Use the saved serial demo binary for the baseline and the current binary for
the candidate, on separate local ports. Do not run their inference concurrently.
All commands, servers and browser children use affinity 0–7.

```sh
taskset -c 0-7 python3 scripts/qa_live_browser.py \
  --url http://127.0.0.1:8099 \
  --video generated/live-demo-qa/camera.y4m --detect-interval 5 --poses 300 \
  --nsys /path/to/nsight-systems-cli/version \
  --output generated/live-pipeline-20260921/browser-adaptive
# Baseline: add --serial, select the baseline port and a separate output folder.

taskset -c 0-7 python3 scripts/qa_live_pipeline.py \
  --frames generated/reference/e2e/audit-20260918/manifest2/frames \
  --detect-interval 5 --output /tmp/live-parity.json
# Repeat at --detect-interval 1 for upstream cadence.

taskset -c 0-7 env \
  LD_PRELOAD="$PWD/build/vulkan/libgemx-profile-ggml.so" \
  GEMX_GGML_TRACE=/path/to/transfers.csv \
  python3 scripts/profile_live_worker.py \
  --frames generated/reference/e2e/audit-20260918/manifest2/frames \
  --detect-interval 5 --repeats 2 --output /path/to/new-native-profile
```

The optional counter collection uses the existing `gemx-reference:e2e` container
and requires GPU-counter access. `scripts/summarize_live_pipeline.py` regenerates
the compact report from the raw run directory. CPU API tracing is for isolated
profiling only; it is not enabled in the deployed demo.
