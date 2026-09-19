# Live profiling and parity-preserving optimization — 2026-09-19

The optimized resident worker takes **122.11 ms per frame**, down from
**125.65 ms**: a **2.82% reduction**, or 7.96 to 8.19 worker frames/s.
Every emitted pose remains byte-identical in the measured replay. This is a
modest improvement; the detector and ViTPose networks still dominate.

## Measurement

RTX 5070 Ti, Ryzen 9 7900, CPU affinity 0–7, eight inference threads and at most
eight build jobs per build. Strict F32 remains enabled: Vulkan F16 and both
cooperative-matrix paths are disabled. Inputs are the public, pinned 480×270
football exemplar, repeated three times without dropping frames. The rolling
window remains 30 frames and Body features remain absent.

Two baseline and two final runs each process 216 frames and emit 215 poses.
The table averages each pair's steady-state means, with 187 full-window frames
per run; the first 29 frames are reported separately in the raw profiles.
Model startup is approximately one second. Timings include synchronization,
CPU preprocessing and transfers within each stage, not just GPU kernel time.

| Stage | Before, ms | After, ms |
| --- | ---: | ---: |
| Read packed image | 0.044 | 0.038 |
| YOLOX preprocessing/detection | 58.428 | 55.317 |
| ViTPose preparation/inference/decode | 62.619 | 62.464 |
| GEM inference and motion decode | 3.709 | 3.703 |
| World skeleton | 0.354 | 0.350 |
| Camera skeleton | 0.270 | 0.007 |
| Write pose | 0.219 | 0.222 |
| **Total** | **125.652** | **122.110** |

The HTTPS devroute replay separately produced all 71 reference poses with
zero keypoint or joint-position difference. Mean request time was 135.61 ms,
p95 137.48 ms: 122.25 ms native-worker time and 13.36 ms remaining request cost.
That remainder includes upload, image decoding/packing, TLS/HTTP and response
transport. The test client opens requests separately; this is not a physical
camera or remote user's latency measurement.

[Machine-readable results](live-profile-2026-09-19.json) include per-run-summary
medians/p95, differential preprocessing checks and the pose-hash-list digest.
Local CSVs, complete pose hashes and logs are under
`generated/live-profile-20260919/`.

## Implemented changes

- YOLOX resize computes OpenCV interpolation coefficients once per coordinate
  instead of repeatedly per pixel/channel/Focus quadrant. Writes now traverse
  each output channel contiguously. Integer rounding and operation order within
  each pixel are unchanged. At 960×540 input, isolated preprocessing median
  falls from 5.76 to 2.80 ms; 25 sizes, including padded rows and one-pixel axes,
  produce identical output bytes and resize ratios.
- Single-frame ViTPose preprocessing runs directly instead of creating and
  joining an async thread on every camera frame. Multi-frame offline batches
  retain parallel crop preparation. The crop and model math are unchanged.
- Skeleton construction retains one exact body-shape fit per session. World
  and camera views of the same prediction share it. Cache keys compare all
  averaged identity and scale bytes; changes invalidate the fit. The cache is
  bounded and guarded by the existing session mutex. No shape smoothing,
  quantization or approximate comparison is introduced.
- Optional `GEMX_LIVE_PROFILE=/path/stages.csv` records native worker stages.
  Profiling is off by default and does not change the worker protocol.

## Parity checks

- All 215 complete `GEMPOSE2` outputs match baseline bytes in each final run,
  covering warm-up, eviction and repeat boundaries. This includes both skeleton
  views, rotations, offsets, camera translation and observations.
- The independent 72-frame HTTPS replay matches accepted native live results.
- Existing OpenCV resize/crop fixtures and release/Vulkan CTest suites pass.
- Offline and live network/decoder fixtures pass on CPU and strict Vulkan.
  Offline reference tests also check cache reuse and invalidation by changing
  identity, global scale and local scales, then restoring the original inputs.
- Contact refinement fixture and Go race tests pass.
- The HTTPS simulated-camera browser check passes live rendering, stop/restart,
  offline switching and permission-error recovery, with one request in flight
  and no browser exceptions. Measured display feedback is about 7.4 fps and
  131 ms frame age; this is not a physical webcam measurement.

This preserves the previously documented upstream agreement and its residual
errors; it does not establish bit-exact upstream end-to-end parity beyond the
existing evidence in [the parity report](LIVE-OFFLINE-PARITY.md).

## Remaining optimization targets

The subsequent [ViTPose operation profile](VITPOSE-PROFILING.md) identifies
feed-forward F32 matrix multiplications as its largest cost and distinguishes
GPU activity, arithmetic throughput and host completion polling.

1. **Detector and ViTPose execution:** together about 117.8 ms, over 96% of
   native time. Next useful work is Vulkan per-operation profiling of F32
   convolution/im2col, attention and matrix multiplication. Require unchanged
   outputs before accepting any kernel or fusion change. Current stage timing
   does not establish which individual kernels dominate.
2. **Capture and transport:** measure on the user's actual camera/network.
   Browser image encoding, decode/packing and round-trip delay can matter,
   especially at higher resolutions or over a WAN. Any overlap must retain
   bounded frame age and preserve the accepted observation sequence.
3. **GEM and file I/O:** low priority at roughly 3.7 ms and 0.26 ms respectively.
   Replacing temporary files with a binary pipe would add protocol complexity
   for a small measured gain here.

Reduced precision, less frequent detection, removing flip augmentation,
smaller model inputs and changing the temporal window would affect parity;
they are outside this optimization pass.

## Reproduce

Keep the baseline executable **and** its `libgemx.so` before rebuilding. Use
`--binary` and `--library-dir` to select that preserved pair. Run benchmarks
serially, with no competing inference or compilation.

```sh
taskset -c 0-7 python scripts/profile_live_worker.py \
  --frames generated/reference/e2e/audit-20260918/manifest2/frames \
  --output /path/to/new-baseline-run --repeats 3

taskset -c 0-7 python scripts/profile_live_worker.py \
  --frames generated/reference/e2e/audit-20260918/manifest2/frames \
  --output /path/to/new-optimized-run --repeats 3 \
  --compare /path/to/new-baseline-run/report.json

taskset -c 0-7 env OPENBLAS_NUM_THREADS=8 python scripts/profile_yolox_preprocess.py \
  --baseline /path/to/baseline/libgemx.so --candidate build/vulkan/libgemx.so \
  --output /path/to/preprocessing.json
```

`profile_yolox_preprocess.py` requires NumPy. `qa_live_http.py --help` and
`qa_live_browser.py --help` describe the transport and simulated-camera checks.
Use the system CA bundle when a Python virtual environment lacks it; keep TLS
certificate verification enabled.
