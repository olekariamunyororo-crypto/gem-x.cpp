# Strict-FP32 ViTPose fusion and projection experiments — 2026-09-20

Update 2026-09-21: the validated winners are now [enabled by default](VITPOSE-DEFAULTS.md). The measurements and opt-in descriptions below record the original experiments.

The best combination reaches **49.55 ms**, down from bracketing combined-matrix
baselines of **52.14 and 52.54 ms**. This is about 5% faster, but **does not reach
the 41 ms target** from the independent ORT CUDA reference with TF32 disabled.
No backend or precision change is introduced.

The retained combination is fused normalization/scale/bias, fused SiLU gating,
and a 64×64 output tile with 64 threads for the expansion projections.
Each final profile uses ten warm-ups and 200 uninstrumented timed inferences.
Separate GPU timestamps cover 30 instrumented inferences per configuration.
Both configurations draw approximately 300 W during the timed region.

| GPU timestamp group | Before | Combined |
| --- | ---: | ---: |
| Expansion projections, 64 operations | 18.31 ms | 16.66 ms |
| Down projections, 32 operations | 9.69 ms | 9.58 ms |
| QKV projections, 32 operations | 7.88 ms | 7.95 ms |
| Attention-output projections, 32 operations | 2.50 ms | 2.52 ms |

Executed operation groups fall from 1,433 to 1,271: normalization removes
130 operations, and SwiGLU removes 32. These are backend profiler operation
counts, not a claim that every operation corresponds to exactly one GPU kernel.
The four projection groups still take about 36.7 ms, so removing small
operations alone will not close the remaining gap to CUDA.

## Implemented candidates

- `GEMX_VITPOSE_SWIGLU=1` replaces separate SiLU and multiplication with
  GGML's existing split-input SwiGLU operator, removing 32 graph operations.
- `GEMX_VITPOSE_NORM=1` fuses normalization, channel scale and bias into one
  Vulkan shader. Matching requires contiguous, aligned F32 tensors with the
  live shape (1280 channels, 197 tokens, batch two), one-dimensional scale
  and bias, a private intermediate chain and the validated GPU/precision.
  GGML's existing overlap checks remain active. Other shapes use the original
  graph operations. The shader retains the original 512-lane reduction tree.
- `GEMX_VITPOSE_RECT_GROUP=up|qkv|proj|down` and
  `GEMX_VITPOSE_RECT=...` select experimental full-F32 output tiles for
  combined projection matrices. `GEMX_VITPOSE_RECT_BK=32|64` optionally changes
  reduction chunk size for the supported tile sizes. Down projections retain
  the existing split-K selection when the standard tile overrides are enabled.

All remain opt-in. The demo defaults are unchanged. The Vulkan additions are
restricted to the RTX 5070 Ti with F16 and cooperative matrices disabled.
The pinned GGML submodule remains pristine: the existing CMake adapter modifies
only a generated backend source, and compiles the new shader with `glslc`.

## Precision investigation

The first normalization shader used `precise` on affine results. GLSL propagated
NoContraction annotations backwards through the reduction, changing its original
arithmetic. The synthetic heatmap error was small (maximum 2.53e-7, mean 3.82e-9,
zero changed peaks), but it failed the byte-for-byte check and was rejected.
The revised shader separates normalization statistics and normalization arithmetic
into functions so only the final floating-point scale and bias are marked
NoContraction. This preserves separate rounding without changing the original
reduction's contraction behaviour.

## Measurement scope

Screening uses fresh processes, ten warm-ups, 80 timed calls, a deterministic
normalized input, and heatmap SHA-256 checks. All compilation and runtime use
CPU affinity 0–7; compilation uses at most eight jobs. GPU runs are sequential.
Screening has measurable drift between opening and closing baselines, so small
ranking differences alone do not establish wins. Final comparisons must use
longer, bracketed runs and actual real-crop and full-pose parity checks.

## Final validation

All 72 real-video crops and flips produce byte-identical heatmaps against the
accepted baseline at batch two and batch eight. Complete live replay processes
216 frames, producing 215 poses, all byte-identical. Detection runs every frame.
Over the 187 full-window frames, ViTPose including crop/decode improves from
**51.956 to 49.213 ms**, and the complete worker from **111.695 to 109.187 ms**
(2.2% lower latency). Camera, browser and transport costs are excluded.

The upstream fixture passes with the same errors as the accepted baseline:
maximum heatmap error 9.57167e-7, mean 1.46851e-8; maximum keypoint error
3.57628e-7; zero pixel displacement. All five CTest contracts, Python syntax
checks and whitespace checks pass. This establishes parity for the tested
fixtures and clips on the validated device, not exhaustively for all inputs.

Raw artifacts are under `generated/vitpose-fusions-20260920/`.
`scripts/experiment_vitpose_fusions.py --output PATH` reproduces the screening
cases; `--cases NAME...` restricts the sweep while retaining its baseline.

The committed [measurement summary](vitpose-fusions-2026-09-20.json) includes
screening results, profiles, telemetry and parity evidence. The final source
implements the corrected normalization shader; the two earlier screening
reports record the rejected versions, so their normalization hashes differ.

To enable the accepted combination in a new worker process, set:

```sh
GEMX_VITPOSE_FLATTEN=1
GEMX_VITPOSE_SWIGLU=1
GEMX_VITPOSE_NORM=1
GEMX_VITPOSE_RECT_GROUP=up
GEMX_VITPOSE_RECT=64x64
```

These must be exported or supplied via `env`. Runtime precision flags remain
`GGML_VK_DISABLE_F16=1`, `GGML_VK_DISABLE_COOPMAT=1`, and
`GGML_VK_DISABLE_COOPMAT2=1`. The profile/screening scripts set those flags.
For a short reproducible comparison:

```sh
taskset -c 0-7 python scripts/experiment_vitpose_fusions.py \
  --output generated/my-fusions --iterations 200 \
  --cases combined baseline-repeat
```
