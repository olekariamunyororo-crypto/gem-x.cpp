# Combining crop and flip projection matrices — 2026-09-20

Combining the two 197-token projection matrices into one 394-token matrix
reduces strict-F32 ViTPose inference from **54.450 to 52.001 ms (4.5%)** on
the RTX 5070 Ti, provided the existing output-tile tuning is retained.
The straightforward reshape with stock tile selection instead regresses to
63.835 ms. This is a useful but modest gain, not an explanation for the entire
CUDA performance advantage.

## Implementation and scope

Set `GEMX_VITPOSE_FLATTEN=1` before starting the worker or benchmark. This is
an opt-in experiment; the demo retains its current default. Unset the variable
or set it to `0` to use the existing graph.

The linear helper reshapes contiguous batch-two inputs to two dimensions,
performs the projection, and restores the original token/batch dimensions
before adding bias. These reshapes are views, with no activation copy.
QKV, attention output, both feed-forward expansion projections and the down
projection use this path. Attention heads remain independently batched.
Other batch sizes and non-contiguous inputs retain their original path.

The Vulkan adapter recognizes the combined shape and retains the existing
32×32 expansion/output tiles and 128×32 down tile. The down projection still
uses three split-K partitions on this device. No dot-product arithmetic,
precision, model weights or backend is changed. The tuned selection remains
restricted to the RTX 5070 Ti and strict F32 without cooperative matrices.
Other devices and precisions have not been benchmarked with this experiment.

## Measurements

All runs use CPU affinity 0–7, ten warm-ups, sequential GPU execution, and
disabled F16/cooperative matrices. Build parallelism is capped at eight.

| Configuration | Timed inferences | Mean inference |
| --- | ---: | ---: |
| Baseline before initial experiment | 120 | 54.125 ms |
| Combined matrices, stock tile selection | 120 | 63.835 ms |
| Baseline repeat | 120 | 54.389 ms |
| Combined matrices, retained tile tuning | 200 | 52.001 ms |
| Paired baseline | 200 | 54.450 ms |

Separate GPU timestamp captures use 30 inferences. Most of the gain comes
from the 64 feed-forward expansion operations: their aggregate time falls
from approximately 20.3 to 18.4 ms. Down projections remain approximately
9.7 ms, QKV approximately 7.9 ms and attention-output projections approximately
2.5 ms. Combining the rows reduces expansion column tile capacity from
2×224 to 416 for 394 useful columns, which plausibly explains part of the
gain. This experiment does not isolate tile-edge waste from scheduling or
cache effects and does not measure a change in occupancy.

## Parity and live replay

Paired live replays measure **54.016 → 51.927 ms** for ViTPose and
**113.754 → 111.674 ms** for the complete worker (1.8% lower total latency).
Steady-state statistics cover 187 full-window frames and exclude camera,
browser and transport costs. Detection runs every frame in both replays.

- Synthetic benchmark heatmaps are byte-identical in all five runs.
- All 72 real-video crops and flips match the accepted baseline byte-for-byte
  at batch two and batch eight. Batch eight is intentionally unchanged.
- Live replay processes 216 frames and produces 215 complete poses, all
  byte-identical to the accepted baseline, with detection every frame.
- The upstream fixture passes: maximum heatmap error 9.57167e-7, mean
  1.46851e-8; maximum keypoint error 3.57628e-7; zero pixel displacement.
- All five CTest contracts pass. Parity is established for these fixtures
  and clips, not exhaustively for every input or device.

Raw artifacts are under `generated/vitpose-flatten-20260920/`; the committed
[measurement summary](vitpose-flatten-2026-09-20.json) records the timings and
parity results.

## Reproduction

```sh
taskset -c 0-7 env GEMX_VITPOSE_FLATTEN=1 python scripts/profile_vitpose.py \
  --output generated/my-flatten --iterations 200 --runs baseline operators

taskset -c 0-7 env GEMX_VITPOSE_FLATTEN=0 python scripts/profile_vitpose.py \
  --output generated/my-flatten-baseline --iterations 200 --runs baseline operators
```

The profiling helper sets strict F32 flags. To use stock tile selection with
combined matrices, additionally set `GEMX_VITPOSE_TILES=0`.
