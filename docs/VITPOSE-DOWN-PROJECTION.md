# ViTPose down-projection experiment — 2026-09-20

The accepted candidate uses **128×32 output tiles with the original split-K
partitions**. Isolated GPU timestamps show about 10% lower down-projection
time; complete-worker replay shows a smaller, roughly 1% gain because the
detector and other stages are unchanged.

This experiment separates output tiling from split-K selection for the
feed-forward down projection: M=1280, N=197, K=5120, batch two. The baseline
includes the already accepted gate/up and attention-output tile tuning.

## Preserving arithmetic

GGML normally chooses the reduction partition count from the selected output
tile. On this RTX 5070 Ti the original 128×128 tile gives three partitions.
The smaller stock tiles give one, changing floating-point accumulation order.
That explains the previous heatmap parity failures.

The experiment saves the original pipeline before overriding output tiles and
uses it for `ggml_vk_guess_split_k`. The original split size, rounded up to a
multiple of 256, is 1792: the three intervals are [0,1792), [1792,3584), and
[3584,5120). The existing partial-sum reduction kernel is unchanged.

All experiments use strict F32, with F16 and both cooperative-matrix paths
disabled. Each case runs in a fresh process, with ten warm-ups and 120 timed
inferences. CPU affinity is 0–7; compilation is restricted to that affinity.
GPU clocks and power limits are not changed.

## Stock square tiles

| Configuration | Mean inference, ms | Exact synthetic heatmaps |
| --- | ---: | --- |
| Current tuned baseline | 55.064 | Yes |
| Down 64×64, original split-K | 55.579 | Yes |
| Down 32×32, original split-K | 55.684 | Yes |
| Current tuned baseline, repeat | 55.312 | Yes |

Both candidates fix parity but regress latency. They are rejected.

## Rectangular full-F32 tiles

| Output tile / workgroup | Mean inference, ms | Exact synthetic heatmaps |
| --- | ---: | --- |
| Current tuned baseline | 55.306 | Yes |
| 128×64 / 128 threads | 55.698 | Yes |
| 64×128 / 128 threads | 55.191 | Yes |
| 64×32 / 64 threads | 54.899 | Yes |
| **128×32 / 128 threads** | **54.628** | **Yes** |
| 32×64 / 64 threads | 56.851 | Yes |
| 64×64 / 64 threads | 55.606 | Yes |
| Current tuned baseline, repeat | 55.500 | Yes |

A first rectangular harness used the wrong shader blob (the default blob
uses F16 intermediates). Those runs are excluded. The corrected sweep uses
`matmul_f32_f32_fp32` explicitly, as does the production integration.

Independent 200-iteration profiles bracket the candidate with the previous
selection. Their uninstrumented mean inference times are 55.334, **54.468**,
and 55.628 ms. Thirty timestamp-instrumented inferences per variant measure
the 32 down-projection operations at 10.940, **9.861**, and 10.941 ms.
GPU power remains approximately 300 W for all three runs.

The winning tile retains 128 output rows but reduces output columns from 128
to 32. At N=197 this reduces column capacity from 256 to 224, with more
workgroups and fewer accumulators per thread. These are plausible contributors
to the gain; this experiment does not isolate their individual effects or
measure per-projection occupancy.

## Real-video validation

All heatmaps from the 72 normalized real-video crops and flips match the
accepted pre-tuning baseline byte-for-byte at batch two and batch eight.
Complete live replay covers 216 frames (the pinned 72-frame clip repeated
three times), producing 215 complete poses, all byte-identical to the accepted
baseline. Steady-state timings cover 187 full-window frames:

| Isolated experimental backend | Previous selection | 128×32 down |
| --- | ---: | ---: |
| ViTPose including crop/decode | 55.303 ms | 54.236 ms |
| Complete native worker | 115.030 ms | 114.191 ms |

These exclude browser/camera/transport costs. Parity is established on the
tested fixtures and clips, not exhaustively for every possible input.

## Integration

The existing CMake adapter inserts a dedicated lazy pipeline into a generated
copy of the backend. It uses the pinned full-F32 shader with new specialization
constants. No upstream GGML files are modified. Selection is restricted to
the RTX 5070 Ti, strict F32 without cooperative matrices, aligned inputs, and
M=1280/N=197/K=5120/batch two. Other shapes and offline batch eight keep their
previous selection.

`GEMX_VITPOSE_DOWN_TILES=0` disables only this change.
`GEMX_VITPOSE_TILES=0` disables all ViTPose tile overrides.
`GEMX_VITPOSE_TILES_TRACE=1` logs activation and preservation of original split-K.
New demo worker sessions load the rebuilt backend; existing workers retain
their loaded library until restarted.

The rebuilt production module independently measures **54.508 ms** versus
**55.606 ms** with only the down override disabled (200 iterations each).
Its down-projection timestamp total is **9.791 ms** versus **10.947 ms**.
Both output hashes equal the accepted baseline. This confirms that the gain
survives integration, rather than depending on the experimental selector.

Final production-worker replay, with the down override enabled versus disabled,
measures **54.183 vs 55.412 ms** for ViTPose and **114.095 vs 115.387 ms** for
the complete worker (about 1.1% lower total latency). Both replays produce all
215 poses byte-identical to the accepted baseline. Activation logs show the
down override only in the enabled run. Real-crop heatmaps also match at both
batch sizes with the integrated override enabled and disabled.

The upstream ViTPose fixture passes: maximum heatmap error 9.57167e-7, mean
1.46851e-8; maximum keypoint error 3.57628e-7; zero pixel displacement. All four
CTest contracts pass, as do Python syntax and whitespace checks.

Driver compiler statistics for the new shader report 128 registers, 21,760
bytes of shared memory and a 34,688-byte binary, versus the earlier large
shader's 255 registers, 49,152 shared bytes and 121,216-byte binary. These
support a reduced resource footprint; they are not measured occupancy.
The invalid-looking local-memory-size statistic is not used to infer spills.

## Reproduction

The experiment builds an isolated backend from the pinned upstream source and
reuses existing shader objects. It does not overwrite the production backend
or modify the GGML submodule. `--preserve-split-k` separates tile selection
from reduction partitioning. `--custom-down-tiles` additionally creates a
dedicated experimental scalar pipeline with rectangular specialization
constants, selected only for the batch-two down-projection shape.

```sh
taskset -c 0-7 python scripts/experiment_vitpose_tiles.py \
  --output generated/my-down-square --preserve-split-k --iterations 120 \
  --cases up-proj-s up-proj-s-down-m up-proj-s-down-s up-proj-s-repeat

taskset -c 0-7 python scripts/experiment_vitpose_tiles.py \
  --output generated/my-down-rectangles --custom-down-tiles --iterations 120
```

`--skip-build` checks the recorded build options before reusing a library.
Raw outputs and timings are under `generated/vitpose-down-20260920/`.
The committed [measurement summary](vitpose-down-2026-09-20.json) contains
sweep results, operator profiles, live timings, parity hashes and provenance.
