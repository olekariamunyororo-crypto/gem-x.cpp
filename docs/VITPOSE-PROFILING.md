# ViTPose GPU profile — 2026-09-19

ViTPose is primarily limited by its F32 matrix multiplications. A live-sized
inference takes **62.06 ms**, with **98–99% GPU activity**. Feed-forward matrix
multiplications consume **57.1%** of the measured GPU span. Transfers are small;
the most evident host inefficiency is CPU polling while waiting for the GPU.

## Measurement and scope

RTX 5070 Ti, NVIDIA driver 595.71.05, Ryzen 9 7900, affinity 0–7, eight inference
threads. F16, cooperative matrices and cooperative matrices 2 remain disabled.
The production inference code and running devroute demo were not changed.

The normalized benchmark uses deterministic synthetic data with the real
network shapes: 197 tokens, width 1280, 32 transformer blocks, 20 heads and
5120-wide feed-forward layers. **Batch 2 corresponds to one camera crop plus
its flip**; batch 8 corresponds to the existing offline four-frame batch.
This isolates network execution, excluding detection, RGB cropping, heatmap
decoding, transport and camera capture. It is not a new video-quality test.

There are ten warm-ups per run, 200 timed calls per baseline/host run, 30 per
GPU timestamp mode, 200/100/50 for batches 1/4/8, and a separate two-call
dependency-barrier diagnostic. Baselines bracket the diagnostic runs. Their
means are 62.032 and 62.094 ms; p95 values are 62.377 and 62.582 ms. The largest
observed baseline call is 66.282 ms. An earlier full run independently measured
62.044 and 61.802 ms.

[Machine-readable results](vitpose-profile-2026-09-19.json) contain complete
operator summaries, telemetry, host spans, hashes and revisions. Raw logs/CSVs
are locally under `generated/vitpose-profile-20260919/{final,offline}/`.

## Expensive GPU operations

GGML Vulkan timestamps, averaged over 30 warmed calls. Percentages use the
**62.864 ms instrumented GPU span**, not uninstrumented wall time. Each row
aggregates all occurrences in one batch-2 inference.

| Operation | Count | GPU ms | Share |
| --- | ---: | ---: | ---: |
| Feed-forward gate/up: 1280 → 5120 | 64 | 24.994 | 39.8% |
| Feed-forward down: 5120 → 1280 | 32 | 10.897 | 17.3% |
| QKV projection: 1280 → 3840 | 32 | 7.861 | 12.5% |
| Attention output projection: 1280 → 1280 | 32 | 5.081 | 8.1% |
| Elementwise ADD | 357 | 3.115 | 5.0% |
| Elementwise MUL | 289 | 1.974 | 3.1% |
| Attention QK matrix multiplication | 32 | 1.876 | 3.0% |
| Attention probability × V | 32 | 1.702 | 2.7% |
| Two transposed convolutions in the heatmap head | 2 | 2.078 | 3.3% |
| Contiguous copies/layout conversions | 229 | 0.968 | 1.5% |
| All remaining operations | 332 | 2.320 | 3.7% |

All matrix multiplies together take **52.544 ms / 83.6%**. Softmax alone is only
0.394 ms, normalization 0.407 ms, and patch im2col 0.006 ms. Attention's QK/V
work is much smaller than the feed-forward and projection work; an attention
rewrite would not address the largest cost.

The logger inserts timestamp queries and additional barriers and changes the
final fence wait. Its wall time rises to 64.286 ms (**3.6% overhead**). The
concurrent logger, which groups operations at existing dependency boundaries,
gives a similar 62.818 ms GPU span and the same ranking. Its implementation
omits the final tail after the last dependency timestamp. Neither mode measures
pure shader execution independently of barriers, queue bubbles and scheduling.

## GPU usage versus arithmetic efficiency

During the two baseline runs, after excluding the first two seconds of telemetry:

- GPU activity averages 98.38% and 98.66%; SM clocks average 2.813 and 2.808 GHz.
- Power averages 255.9 and 258.9 W against a 300 W limit. Power-cap and hardware
  thermal-slowdown flags stay inactive; temperature stays between 57 and 62°C.
- Memory activity averages 9.43% and 9.66%. This is **not** bandwidth saturation.
- Device-wide memory use is 12,437 MiB (12.15 GiB), including
  other resident models; it is not ViTPose's allocation alone.

NVIDIA defines these utilization counters as time with GPU work or memory
accesses during the sampling window. They do not measure active warps, occupied
SMs, achieved bandwidth, or instruction throughput.
[NVIDIA utilization documentation](https://docs.nvidia.com/deploy/nvidia-smi/).

Useful matrix arithmetic totals **674.46 GFLOP per inference**, calculated from
the logged shapes and counts as `M*N*(2*K-1)*batch`. Dividing by matrix operation
time gives **12.84 TFLOP/s**; dividing by baseline inference time gives
**10.87 TFLOP/s** for matrix arithmetic alone. The backend's transposed-convolution
FLOP estimates are excluded because they count a dense output/kernel product
that overstates the useful stride-2 arithmetic.

For scale, 8,960 FP32 lanes × two operations per FMA × the observed clock gives
an ideal **50.37 TFLOP/s** ceiling. Thus useful matrix throughput is approximately
**25.5% of that ideal peak**, despite near-continuous GPU activity. This is a
calculated arithmetic ratio, **not measured occupancy or a predicted speedup**.
[NVIDIA RTX 5070 Ti specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/).

Nsight Systems/Compute were unavailable. Register pressure, cache misses,
warp-stall reasons, SM occupancy and precise queue-idle intervals have not been
measured. Memory activity alone cannot establish whether individual kernels
are bandwidth-bound.

## Synchronization and stalls

An optional LD_PRELOAD tracer measures GGML host calls without adding GPU
barriers. All 200 inferences contain exactly one of each call below.

| Host operation | Wall ms | Calling-thread CPU ms |
| --- | ---: | ---: |
| Upload 1,179,648 bytes | 0.055 | 0.054 |
| Record/submit graph | 2.203 | 2.135 |
| Synchronize graph | 59.014 | 12.899 |
| Download 1,892,352 bytes | 0.242 | 0.143 |

Total instrumented wall time is 61.877 ms; the remaining approximately 0.362 ms
covers validation, graph lookup/support checks and harness/tracer overhead.
GPU work overlaps host command recording/submission. The synchronization span
is the **CPU waiting for completion**, not 59 ms of GPU inactivity.

Source inspection in `ggml_vk_wait_for_fence` shows a blocking wait on an
"almost ready" fence followed by repeated `getFenceStatus` polling with CPU
pause instructions. The 12.9 ms of CPU time in synchronization is consistent
with that busy-wait tail. Overall, inference consumes about 15.8 ms of process
CPU per 62 ms wall time—roughly 25% of one core, far below the eight-core limit.
These measurements do not isolate driver CPU work from the polling itself.

The synchronization logger counts **1,433 executed graph operations and 1,300
dependency barriers per inference**. Barriers are GPU memory dependencies,
not 1,300 host fence waits, and an operation may contain multiple dispatches.
There is substantial scope to examine fine-grained operation/fusion overhead,
but the counts alone do not quantify how much time barriers could save.

High device activity, stable inference times and 0.30 ms total transfer API
time provide no evidence of large recurring host/transfer starvation. They do
not rule out short intra-kernel stalls or queue gaps. The optional
`final/host/host-trace.json` can be opened in Perfetto/Chrome's trace viewer;
its tracks are explicitly host API ranges, not a GPU execution timeline.

## Batch scaling

| Normalized batch | Source frames with flip | Call ms | Source frames/s |
| --- | ---: | ---: | ---: |
| 1 | Diagnostic, no flip pair | 41.200 | — |
| 2 | 1, live | 62.063 | 16.11 |
| 4 | 2 | 107.407 | 18.62 |
| 8 | 4, existing offline batch | 200.955 | 19.90 |

Batch 8 reaches 99.04% GPU activity and 300.02 W; its **software power-cap flag
is active**, with 2.776 GHz average SM clock and no thermal-slowdown flag.
Unlike the live batch, the offline batch reaches the power limit. Larger
batches improve throughput modestly but increase latency and require waiting
for frames, so they are not a straightforward live-latency optimization.

## Optimization priorities

1. **F32 GEMM geometry and scheduling.** Prioritize the two feed-forward input
   projections, then the output projection. The generic non-cooperative path
   selects 128×128 large tiles for these shapes; 197 token columns occupy only
   197/256 of two tiles. The 1280×197 attention projection has just 40 such
   workgroups across batch two when unsplit. These are source-level clues for
   tile/parallelism experiments, not hardware-counter proof of the bottleneck.
   Reduction order, split-K and compiler contraction can change output bits,
   so every experiment needs numerical comparison.
2. **Reduce completion polling for CPU efficiency.** A blocking wait or a
   shorter polling tail could recover much of the 12.9 ms CPU cost. Measure
   wake-up latency before adopting it; this does not remove the GPU computation
   or imply a similar wall-time saving.
3. **Fuse selected elementwise/layout operations.** ADD/MUL alone account for
   5.09 ms, with many dependency boundaries. Norm affine, residual/gamma and
   rotary paths deserve targeted experiments. Preserve operation order and
   rounding; removing barriers without proving dependencies is unsafe.
4. **Transfers/preprocessing are lower priority for ViTPose.** Moving 3.07 MB
   across the API costs only about 0.30 ms here. Reduced precision or removing
   flip augmentation would change the parity conditions and was not tested.

No inference optimization was deployed from this investigation. It adds
profiling tools and evidence so the next changes can target measured costs.

## Reproduce and validation

Use the existing Release Vulkan build with sanitizers disabled:

```sh
taskset -c 0-7 cmake --build build/vulkan \
  --target gemx-vitpose-benchmark gemx-profile-ggml -j8
taskset -c 0-7 python scripts/profile_vitpose.py \
  --output generated/vitpose-profile-new
```

`--runs host operators concurrent` selects diagnostic modes; `--runs batch8`
selects the offline-shaped measurement. The script requires NVIDIA telemetry,
refuses affinity wider than eight cores, and always supplies strict F32 flags.
Short selected runs can have no telemetry samples after the two-second exclusion.

The benchmark's optional `GEMX_BENCHMARK_PROFILE`, `GEMX_BENCHMARK_OUTPUT` and
`GEMX_BENCHMARK_WARMUP` variables enable timing CSV, final heatmap output and
warm-up count. `GEMX_GGML_TRACE` controls the Linux preload library. Do not
preload the tracer into the production demo.

Both complete profiling passes produced identical final heatmap bytes across
baseline, host, per-operation, concurrent and repeat modes at batch 2. The
final synchronization diagnostic matched too. SHA-256:
`a4d8fbccfe9e3379d7a044b13c632f3855a121591d857f44141274a4fde937cf`.
This compares instrumentation modes on the benchmark input, not batch sizes
or upstream video parity. The build, four existing Vulkan-build CTest contracts,
Python syntax check and whitespace check pass.
