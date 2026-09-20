# Individual QKV counters — 2026-09-20

The isolated QKV investigation identifies **high Vulkan resource use, a small workgroup grid and unusually high instruction-dispatch stalls**. It does not identify VRAM bandwidth or instruction-cache misses as the main problem. The remaining work is to improve the shader's execution efficiency; higher occupancy by itself is not a demonstrated speedup.

## Scope and validation

The profiling-only preload executes the real first-block graph prefix on the pinned batch-two input, then repeatedly executes just its QKV matmul on the resident operands. Dimensions are output M=3840, tokens N=394, reduction K=1280. Bias addition and the rest of ViTPose are excluded. It exits without returning an inference result and is never used by the demo.

The first capture submitted one matmul at a time. The main capture submits 32 repeated matmuls per graph to reduce host gaps. Compute-in-flight rose from 81.73% to 98.00%, while the warp-state distribution barely changed. Only one compute shader appears in these captures. Replay output was checked against 1,081 double-precision dot products spread across the output (maximum absolute error 3.95e-5, mean 1.29e-6; all output values finite). This validates the isolated graph's operands and output, not full-model parity.

CUDA uses the previous first-block QKV kernel capture from the original ONNX graph, with kernel replay and caches left uncontrolled. Vulkan uses Nsight Graphics 2026.3.1; CUDA uses Nsight Compute 2025.1. Both are strict FP32 on the RTX 5070 Ti, with CPU affinity limited to eight cores and clocks unaltered. These now represent the same logical operation, but are **not identical cache histories or collectors**. Resident Vulkan replay particularly understates cold-memory traffic.

## Resources and parallelism

| Property | Vulkan default QKV | CUDA QKV |
|---|---:|---:|
| Output tile | 128×128 | 128×64 |
| Threads per workgroup/block | 128 | 128 |
| Registers per thread | **255** | **128** |
| Shader shared memory per block | **48 KiB** | **30 KiB** |
| Maximum resident blocks from these resources | 2 | 3 |
| Corresponding maximum occupancy | 16.67% | 25.00% |
| Measured occupancy | 12.53%, elapsed-cycle sampling | 22.70%, active-cycle counter |

The driver identifies the Vulkan pipeline as `matmul_f32_f32_aligned_l`. Its compiled binary is 121,216 bytes. The reported local-memory-size value is implausible (`68719476752`), so it is not used to infer register spills. CUDA's allocation additionally includes 1 KiB driver shared memory per block. CUDA is shared-memory limited to three resident blocks; its registers alone would allow four.

Vulkan's grid has `ceil(3840/128) × ceil(394/128) = 120` workgroups on 70 SMs. With four warps per group and 48 maximum warps per SM, even simultaneously resident groups would provide only `120×4/(70×48) = 14.29%` whole-GPU occupancy. Tail effects reduce that further. K=1280 does not activate this backend's K≥2048 split-K heuristic. This explains why QKV occupancy is much lower than Vulkan's previously reported whole-model 22.9%.

CUDA's captured launch grid is `(56,4,2)` with 448 blocks and 2.13 theoretical waves per SM. Its grid must not be interpreted as a direct rectangular tiling: library scheduling/swizzling can differ from the tile name.

## Individual warp states

Percentages are normalized over **all sampled active warp states**, including selected and not-selected. They are not wall-time fractions, idle-SM fractions or guaranteed recoverable time.

| Warp state | Vulkan QKV | CUDA QKV |
|---|---:|---:|
| Dispatch stall | **16.69%** | **4.77%** |
| Short scoreboard | 8.73% | 5.18% |
| MIO throttle | 6.16% | 20.01% |
| Barrier | 3.18% | 11.99% |
| Math dependency wait | 3.56% | 2.50% |
| Long scoreboard | 2.95% | 1.62% |
| No instruction | 2.08% | 3.37% |
| Math-pipeline throttle | 0.44% | 0.48% |
| Selected | 33.79% | 26.23% |
| Not selected | 21.20% | 22.69% |

A dispatch stall means a pipeline interlock prevented a selected warp from dispatching an instruction. It does **not** mean the CPU failed to launch a kernel, nor an instruction-cache miss. NVIDIA's guide specifically flags dispatch shares above 5% for investigation. Our 16.7% persists after reducing CPU submission gaps. The precise interlock still needs instruction-level attribution. [NVIDIA shader-profiler definitions](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html#stall-reasons).

Short scoreboard concerns dependencies on short-latency operations, commonly shared-memory loads. MIO pressure and barrier waits are stronger in CUDA, yet CUDA is faster: removing those categories indiscriminately is not the right objective. Selected and not-selected are scheduler states, not faults. [NVIDIA Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

Vulkan's instruction-cache hit rate is **99.9875%**; its math-pipeline-throttle share is low. The isolated run shows 51.16% SM throughput, 41.07% FMA-pipe throughput and 18.62% L2 throughput. These elapsed-window figures should not be directly equated with CUDA's active-cycle counters. Its near-zero DRAM traffic reflects resident replay, not the complete model's weight traffic.

CUDA QKV has 1.36 eligible warps per scheduler and issues in 68.13% of active cycles. Its L1/TEX throughput is 72.55% of active-cycle peak, with about 15.15 million shared wavefronts versus 10.38 million ideal (46% excess over ideal). That corroborates shared-memory pressure in CUDA, but does not establish the cause of Vulkan's slower execution.

## Why simply shrinking the tile did not help

The earlier full-model sweep already found the existing 128×64 QKV variant slower (52.88 ms versus its 51.90 ms baseline). A new isolated capture explains the tradeoff:

| Vulkan QKV metric | Default 128×128 | Existing 128×64 |
|---|---:|---:|
| Registers/thread | 255 | 183 |
| Shared memory/block | 49,152 bytes | 26,112 bytes |
| Compiled binary | 121,216 bytes | 60,544 bytes |
| Elapsed sampled occupancy | 12.53% | 11.00% |
| Dispatch stall share | 16.69% | 18.36% |
| Long-scoreboard share | 2.95% | 9.88% |
| FMA throughput | 41.07% | 34.43% |
| Isolated host-inclusive time per matmul | 0.2417 ms | 0.2527 ms |

Reducing registers to 183 still does not fit three 128-thread blocks into 65,536 registers, even before allocation rounding. The smaller tile therefore fails to reproduce CUDA's resource balance. It also increases memory-dependency stalls and loses FMA throughput. These isolated timings use the same 32-dispatch submission loop; they are not replacements for in-model GPU timestamps.

The next targeted experiment should change the **register microtile, shared-memory access layout and instruction scheduling together**, preserving FP32 accumulation order. A resource goal of three resident 128-thread blocks requires both lower registers and suitable shared memory, but that is only a screening criterion: measured instruction issue and QKV time must improve. The unusual dispatch stalls warrant inspecting generated instruction ranges or a compiler reproducer. Current counters cannot identify the exact interlock or prove that reducing it recovers 16.7% of execution time.

No optimization or demo default was changed in this investigation.

## Reproduction

Build `gemx-profile-qkv` alongside the Vulkan benchmark with at most eight build workers. Run the existing capture helper under `taskset -c 0-7` with `--optimized --qkv-only --input PATH`, the local `--ngfx` path and required profiler library path. The raw input is the same normalized batch-two fixture used by the full-model comparison. Add `--qkv-tile 128x64` for the diagnostic alternative.

`GEMX_VITPOSE_TILES_TRACE=1` prints the selected QKV pipeline/tile/register count (a zero register count means asynchronous pipeline compilation has not finished; use the driver statistics for the final count). `GGML_VK_PIPELINE_STATS=matmul_f32_f32_aligned_l` prints the driver's full default-pipeline resource statistics. To dump replay operands and result, set `GEMX_QKV_DUMP` to an existing directory when explicitly preloading `build/vulkan/libgemx-profile-qkv.so`. This preload terminates the benchmark after 15 seconds of isolated replay.

`scripts/summarize_qkv_counters.py` combines Nsight Graphics exports and the first QKV row of the previous Nsight Compute wide CSV. [Committed counter data](vitpose-qkv-counters-2026-09-20.json) includes full metric names and units. Local raw reports and logs are under `generated/vitpose-qkv-20260920/`; CUDA source counters remain under `generated/vitpose-cuda-match-20260920/`.

Validation: all five CTest contracts passed, Python helpers compile, and an ordinary full ViTPose run produced heatmaps byte-identical to the preceding matched-input Vulkan baseline.
