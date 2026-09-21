# Projection experiments after FlashAttention — 2026-09-20

Update 2026-09-21: the validated winners are now [enabled by default](VITPOSE-DEFAULTS.md). The measurements and opt-in descriptions below record the original experiments.

**Result: the best QKV candidate reaches about 49.10 ms, a small improvement; CUDA remains faster at 41.20 ms. Packed gate/up is slower and is not recommended.** The strict-FP32 FlashAttention experiment did not improve latency, so this investigation returns to projection kernels. All results use the RTX 5070 Ti, at most eight CPU cores, F16 and cooperative matrices disabled, and the same pinned batch-two normalized input as the CUDA comparison. No new backend is integrated.

## QKV register microtiles

These variants specialize the existing GGML F32 shader. They change threads, warp layout and per-thread output ownership, preserving the dot-product arithmetic. They are separate from the accepted 64×64 feed-forward expansion tile.

| QKV variant | Full ViTPose latency, 80 iterations | Registers/thread | Fixture byte equality |
|---|---:|---:|---|
| Baseline | 49.36 ms | 255 | Yes |
| 128×64, 256 threads | 49.96 ms | 152 | Yes |
| **64×64, 128 threads** | **48.97 ms** | **128** | **Yes** |
| 64×32, 128 threads | 49.82 ms | 146 | Yes |
| 128×32, one register-microtile iteration | 49.25 ms | 127 | Yes |
| 128×128, 256 threads | 51.33 ms | 191 | Yes |
| 128×64, one register-microtile iteration | 50.01 ms | 196 | Yes |
| Baseline repeat | 49.76 ms | 255 | Yes |

Ten warmups precede each run. Timings include input upload and heatmap download. GPU clocks are not fixed; the baseline repeat demonstrates drift. Register count alone does not predict performance: for example, a 256-thread workgroup can consume more total registers despite using fewer per thread.

Select the candidate with `GEMX_VITPOSE_QKV_LAYOUT=64x64-t128`. It only applies to aligned F32 QKV matrices of shape M=3840, N=394, K=1280 on the validated device, with F16/cooperative matrices disabled. Offline batch-eight QKV keeps its existing selection. The ordinary default is unchanged.

## Packed feed-forward projection

`GEMX_VITPOSE_PACK_FFN=1` packs each block's gate and up weights and biases once when loading the model, then computes a single 1280→10240 projection and a contiguous SwiGLU. The two 1280→5120 projections previously read the same normalized activation separately. Existing GGUF files remain compatible.

The prototype retains the original weights and adds approximately **1.56 GiB** of GPU weight storage plus a small amount for biases, and additional startup transfer work. This is an experimental opt-in, not a deployment default. The packed 10240-output projection uses the accepted 64×64 expansion tile when `GEMX_VITPOSE_RECT_GROUP=up` and `GEMX_VITPOSE_RECT=64x64` are enabled.

## Reproduction

`scripts/experiment_vitpose_qkv.py --input RAW_F32 --output DIRECTORY` runs the microtile screen under `taskset -c 0-7`; `--layouts` selects candidates, including `packed-ffn` and `packed-ffn-qkv`. It sets the accepted baseline optimization flags and disables FlashAttention. Reports preserve environment, input hash, output hash and per-case timings. Existing profiling and real-crop comparison tools provide longer timing runs and numerical/keypoint checks.

## Longer runs and initial packing result

With 200 timed iterations, the QKV candidate measured 49.10 ms versus baseline 49.34 ms and repeat 49.70 ms. All outputs were byte-identical. This is a small improvement and remains far from the 41.20 ms CUDA reference; no larger speedup is established by these data.

An initial packed-FFN diagnostic using the backend's default large tile measured 53.25 ms, or 52.96 ms combined with the QKV candidate. Both retained byte-identical fixture heatmaps. This diagnostic did not select the accepted 64×64 expansion tile for the new 10240-output shape; the follow-up explicitly extends that shape selection before deciding whether packing is useful.


## Final packed-tile result

With the tuned 64×64 expansion tile enabled for the packed shape (200 iterations each):

| Case | Mean inference time |
|---|---:|
| Baseline | 49.79 ms |
| Packed gate/up | 50.77 ms |
| Packed gate/up + QKV candidate | 50.16 ms |
| Baseline repeat | 49.69 ms |

All fixture heatmaps remained byte-identical. Packing is rejected as a performance improvement: it is slower and costs additional GPU memory. Its implementation remains an explicitly opt-in diagnostic for reproduction, as do the QKV layout variants. FlashAttention is disabled throughout these projection experiments. No demo defaults or deployment changed.

The QKV candidate is a modest improvement, **not a match for CUDA**. These results do not justify another blind tile-size sweep. Larger gains likely require changes to the shader's memory/compute instruction schedule and data reuse, supported by individual-operation counters. This is an inference from the failed layouts, not a demonstrated optimization.

Raw local reports: `generated/vitpose-qkv-microtiles-20260920/`, `generated/vitpose-projection-combined-20260920/`, and `generated/vitpose-packed-tuned-20260920/`. The first packed experiment uses the default large tile; the final directory uses the corrected tuned expansion selection.

Validation: both the QKV candidate and packed+QKV produced byte-identical heatmaps for all 72 real-video crops at batch sizes two and eight. All five CTest contracts passed; Python helpers compile and `git diff --check` passes. [Committed measurements](vitpose-projection-experiments-2026-09-20.json).
