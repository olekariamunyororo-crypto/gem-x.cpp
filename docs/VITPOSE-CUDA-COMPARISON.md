# Matched ViTPose CUDA measurements — 2026-09-20

CUDA completes the same batch-two input in **41.20 ms**, versus **49.55 ms** for optimized Vulkan (Vulkan is 20.3% slower). CUDA achieves higher FMA-pipeline utilization despite lower occupancy. This supports investigating useful FP32 arithmetic efficiency, instruction mix and scheduling; it does not establish one exclusive cause.

## Conditions

RTX 5070 Ti, driver 595.71.05; at most eight CPU cores throughout. Both consume the pinned normalized crop plus horizontal flip, shape `[2,3,256,192]`, SHA256 `2c77457a659ab0c71e6e6ab0bdb50f3b7fb1ca38352a50719760250be84a5694`.

CUDA is ONNX Runtime 1.23.2, `use_tf32=0`, CUDA graphs and tunable ops disabled. Vulkan uses F32 with F16/cooperative matrices disabled and the accepted flatten, norm, SwiGLU and 64×64 expansion-tile optimizations. No inference implementation changed during this investigation; CUDA remains a reference benchmark.

Uninstrumented latency includes input upload and output download, excludes preprocessing, and averages 200 iterations. Vulkan uses 10 warmups; CUDA uses 20. Hardware sampling and detailed kernel profiling are separate runs.

## Same-collector hardware comparison

Both workloads were sampled independently using Nsight Systems 2026.5.1, `gb20x-top`, 10 kHz, an eight-second window after warmup, trimming 0.5 seconds from each edge. The collector targets a separate sleep process with API tracing disabled: it does not inject into or serialize either inference workload. No clocks or global driver permissions were changed.

| Metric | Vulkan | CUDA |
|---|---:|---:|
| Uninstrumented latency | 49.55 ms | 41.20 ms |
| SM throughput | 52.54% | 53.64% |
| SM issue-stage throughput | 51.96% | 53.63% |
| FMA-pipe throughput | 35.86% | 43.44% |
| Active compute warp occupancy | 22.87% | 20.29% |
| VRAM bandwidth utilization | 11.58% | 16.55% |
| L2 throughput | 34.77% | 30.59% |
| L1/TEX throughput | 43.62% | 46.07% |
| Tensor pipe active | 0% | 0% |

Throughput figures are sampled percentages of sustained peak. Occupancy sums the collector's two compute-warp categories. Exported physical bandwidth/frequency unit labels are ambiguous, so the comparison uses utilization percentages. Neither workload saturates VRAM bandwidth. Neither uses Tensor Cores in this strict-FP32 comparison. Lower CUDA occupancy rules out “more active warps” as the explanation for its advantage.

With the external sampler running, whole-workload averages were 49.96 ms Vulkan and 41.48 ms CUDA, close to the separate uninstrumented measurements. Those averages extend beyond the sampled window.

## Where the time differs

CUDA timings below are actual CUPTI kernel durations averaged across three warmed inferences; Vulkan timings are fresh per-operation GPU timestamps on the identical input. CUDA's seven-GEMM-per-block mapping is inferred from the repeating launch order and model structure, not node-level NVTX annotations.

| Logical operation, all 32 blocks | Vulkan | CUDA |
|---|---:|---:|
| QKV projection | 7.94 ms | 5.43 ms |
| FFN down projection | 9.58 ms | 7.19 ms |
| FFN gate + up projections | 16.66 ms | 15.29 ms |
| Attention scores + values | 3.59 ms | 1.50 ms |
| Attention output projection | 2.52 ms | 2.73 ms |

QKV and down have a combined observed gap of **4.90 ms**; attention's two matrix multiplications add **2.10 ms**. Expansion is now comparatively close, and Vulkan's attention output projection is slightly faster. QKV/down are the strongest remaining timing targets, followed by attention matmuls. These differences are not an additive guaranteed speedup budget: Vulkan operation timestamps can include split-K/reduction dispatches, while CUDA groups classify GEMM kernels.

CUDA launches 1,395 kernels per inference, including 224 GEMMs. Mean summed kernel duration is 37.20 ms, within a 39.92 ms first-to-last-kernel envelope. The roughly 2.72 ms difference includes copies, gaps and other work; it cannot all be attributed to CPU stalls.

## Kernel counters and stalls

Nsight Compute 2025.1 collected the first warmed block's seven GEMMs with kernel replay (19 passes each). QKV, output and down use CUTLASS SIMT SGEMM 128×64 tiles with 128 registers/thread. Gate/up use 256×128 tiles with 210 registers/thread; attention uses smaller CUTLASS/MAGMA kernels. These are FP32 SIMT kernels.

QKV/down achieve roughly 65% SM throughput, 21–23% active-cycle occupancy and 1.27–1.36 eligible warps per scheduler. Gate/up achieve roughly 65% SM throughput, 16.6% occupancy and 72% issue activity. Thus high register use and low occupancy coexist with efficient issue activity.

CUDA QKV/down have substantial MIO and barrier sample shares. Their no-instruction shares are only about 3–4%. Fresh Vulkan whole-graph PC samples show about 2.3% no-instruction, 12.3% long-load scoreboard, 11.8% dispatch, 10.0% math wait and 8.5% short scoreboard. These are **different scopes**: whole-graph Vulkan stall shares must not be ranked directly against individual CUDA kernels. Exact stall attribution still needs isolated Vulkan kernels. Available evidence does not establish instruction fetching as the dominant bottleneck.

A successful CUDA application-range replay was excluded from the matching aggregate comparison because it raised instrumented latency to roughly 53–63 ms. Individual replay counters are diagnostic; replay timings do not replace ordinary latency or CUPTI timings.

## Accuracy

Against the pinned fixture, uninstrumented CUDA heatmaps have maximum absolute error `7.30e-7`, mean `1.03e-8`; Vulkan has maximum `9.57e-7`, mean `1.47e-8`. Both have zero changed raw peaks across 154 joints. This is a fixture check, not a new end-to-end video quality evaluation.

## Reproduction and artifacts

Machine-readable results: [vitpose-cuda-match-2026-09-20.json](vitpose-cuda-match-2026-09-20.json). Raw traces, reports, commands and logs are retained locally under `generated/vitpose-cuda-match-20260920/` (not committed).

The reference profiling helper accepts `latency`, `timeline`, `kernels` and `counters` modes. Model, weights and Python dependencies must already exist; see its `--help`. For example:

```sh
taskset -c 0-7 python scripts/profile_cuda_vitpose.py --mode latency --output generated/new-cuda-match
taskset -c 0-7 python scripts/profile_cuda_vitpose.py --mode timeline --output generated/new-cuda-match
taskset -c 0-7 python scripts/profile_cuda_vitpose.py --mode kernels --output generated/new-cuda-match
```

Create `input.f32` in that output directory from the same normalized fixture with its horizontal flip, then run `sample_vitpose_gpu.py --output ... --nsys ...` under the same affinity. The Nsight directory must contain `target-linux-x64/nsys`. The sampler requires the latency command manifest and built Vulkan benchmark; each output directory is single-use for sampling. Counter containers require permission to access GPU performance counters. The `counters` mode needs recent Nsight Compute and is retained for diagnostics, not the fair aggregate comparison.

`profile_vitpose.py --input` and `capture_vitpose_nsight.py --optimized --input` support the identical raw batch-two input. The benchmark rejects incorrect input lengths.

Follow-up: [individual QKV counters](VITPOSE-QKV-COUNTERS.md) isolate Vulkan QKV and compare its resource use and warp states with CUDA.
