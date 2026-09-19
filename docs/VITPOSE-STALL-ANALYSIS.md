# ViTPose hardware counters — 2026-09-19

Hardware counters indicate limited shader execution efficiency and latency
hiding, rather than saturated external memory bandwidth. Instruction-fetch
waits are measurable but a small part of sampled active warp states. This is
an aggregate diagnosis of the complete ViTPose graph, not a per-matmul roofline.

## Measurement

Nsight Graphics 2026.3.1 GPU Trace with real-time shader profiling, RTX 5070 Ti
(GB203), driver 595.71.05. Captured the original tile selection once and the
accepted tile selection twice, sequentially. The workload is the existing
ViTPose benchmark with a crop and flip (batch 2), strict F32, 20 warm-ups,
500 requested iterations and capture triggered after four seconds. Each
capture requests 250 ms; trace boundaries can extend the recorded interval.
The profiler terminates the benchmark after export. CPU affinity is 0–7.
Clocks remain unaltered; administrator access enables counters without
changing global driver permissions. These are instrumented captures: retain
the separate uninstrumented benchmarks for latency comparisons.

| Counter, averaged over capture | Original tiles | Tuned | Tuned repeat |
| --- | ---: | ---: | ---: |
| DRAM throughput, % sustained peak | 8.08 | 9.14 | 9.09 |
| L2 throughput, % sustained peak | 17.95 | 36.90 | 37.06 |
| L1/TEX throughput, % sustained peak | 29.93 | 41.91 | 42.10 |
| SM throughput, % sustained peak | 42.42 | 50.46 | 50.69 |
| Active compute warps, % maximum | 16.19 | 21.07 | 21.15 |

SM throughput is Nsight's composite utilization metric, not a measurement of
FP32 FLOP/s or occupancy. The active-warp counter is the occupancy-related
measurement: about 7.77 active warps originally versus 10.11–10.15 after tuning,
against 48 possible. The tuned L2 sector hit rate is about 94%.

Neither DRAM nor the cache throughput averages approach saturation. Low DRAM
traffic does **not** mean memory latency is free: dependent loads can stall
warps even while bandwidth is available. The low active-warp count leaves
less independent work to cover those waits. These averages cannot exclude
a bandwidth limit in an individual short operation.

## What the warps are doing

Percentages below divide each sampled state by the sum of **all** active-warp
states, including selected and not-selected. They are not percentages of
wall-clock time, and do not estimate the speedup from removing a stall.

| Sampled state | Tuned share | Meaning |
| --- | ---: | --- |
| Selected | 20.9% | Warp issued an instruction |
| Not selected | 22.3% | Eligible warp; another warp was scheduled |
| Long scoreboard, L1/TEX | 11.5% | Waiting for a load result |
| Wait | 11.1% | Coupled math dependency |
| Dispatch stall | 10.1% | Pipeline interlock blocks dispatch |
| Short scoreboard | 9.1% | Shorter dependency, including shared-memory/MIO work |
| MIO throttle | 5.1% | MIO instruction queue pressure |
| No instruction | 2.5% | Instruction-fetch/availability wait |
| Math pipe throttle | 2.3% | Math instruction queue pressure |
| Miscellaneous | 2.0% | Other sampled waits |
| Branch resolving | 1.4% | Branch target unresolved |
| Barrier | 1.2% | Waiting at a barrier |
| Local/global throttle | 0.5% | Local/global instruction queue pressure |

Definitions follow NVIDIA's [Shader Profiler documentation](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).
Selected and not-selected are not inherently problematic stalls. Long
scoreboard does not identify DRAM specifically. The no-instruction category
does not establish an instruction-cache miss by itself; see the
[Nsight Compute profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).

The repeat gives essentially the same mix. No-instruction is 2.53–2.54% of
sampled states, versus 2.84% originally. There is no evidence here that
instruction fetching is the dominant limiter. Dispatch interlocks and
dependencies warrant more attention, but these counters do not identify a
specific producer instruction or establish a driver defect.

## Why smaller tiles helped

Vulkan pipeline executable statistics report these compiler resource costs:

| Strict-F32 aligned matmul shader | Original large tile | Accepted small tile |
| --- | ---: | ---: |
| Registers | 255 | 135 |
| Shared memory, bytes | 49,152 | 8,704 |
| Shader binary size, bytes | 121,216 | 35,584 |

The smaller shader has substantially lower resource requirements. Combined
with more independently schedulable workgroups and less padding at 197 tokens,
this is consistent with the measured increase in active warps and SM
throughput. It does not isolate the contribution of each change. The compiler's
reported local-memory-size statistic is implausibly large; it is deliberately
not used to infer spills.

The [earlier tile experiment](VITPOSE-TILE-TUNING.md) measured roughly 12%
lower inference latency with byte-identical outputs. The hardware measurements
now substantiate improved occupancy and resource utilization, which were
previously only hypotheses. No inference implementation changes were made
for this investigation.

## Scope and next optimization targets

The exported GPU Trace region table is empty for this headless benchmark.
Consequently, these counters cannot distinguish gate/up, down, QKV and
attention-output matmuls. Matmuls dominate the prior timestamp profile, but
attributing every sampled stall to them would be incorrect.

The next useful experiment is isolated dispatch capture for each projection
shape, followed by shader-level attribution of load dependencies and dispatch
interlocks. Lower register pressure, shorter dependency chains and more
independent work are better-supported targets than reducing VRAM traffic or
instruction-cache size. Any kernel change still needs the existing exact
heatmap parity checks; down-projection tile changes previously failed that
requirement.

## Reproduction and artifacts

`scripts/capture_vitpose_nsight.py` captures the current backend; `--baseline`
sets `GEMX_VITPOSE_TILES=0`. Use a locally installed Nsight Graphics CLI:

```sh
taskset -c 0-7 python scripts/capture_vitpose_nsight.py \
  --ngfx /path/to/ngfx --sudo --output generated/my-vitpose-trace
```

On NixOS, `--library-path` can provide the tool's runtime dependencies.
`--sudo` requires already configured noninteractive sudo. No GPU clock or
driver permission setting is changed. The script targets this repository's
strict-F32 RTX 5070 Ti experiment and expects its existing Vulkan build/model.

Full exported numeric counters and provenance are committed in
[vitpose-stalls-2026-09-19.json](vitpose-stalls-2026-09-19.json). Regenerate with:

```sh
taskset -c 0-7 python scripts/summarize_vitpose_nsight.py \
  generated/vitpose-stalls-20260919/trace-baseline \
  generated/vitpose-stalls-20260919/trace-tuned \
  generated/vitpose-stalls-20260919/trace-tuned-repeat \
  --output docs/vitpose-stalls-2026-09-19.json
```

The summarizer verifies that the sum of sampled states agrees with the active
warp counter. Raw GPU traces, TSV exports, pipeline statistics and device
attributes remain locally under `generated/vitpose-stalls-20260919/`.
