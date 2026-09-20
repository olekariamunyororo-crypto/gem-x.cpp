# ViTPose F32 tile tuning — 2026-09-19

Follow-up: [hardware counters and warp-stall analysis](VITPOSE-STALL-ANALYSIS.md)
measure the occupancy improvement and distinguish bandwidth from latency limits.
The subsequent [down-projection experiment](VITPOSE-DOWN-PROJECTION.md) adds
128×32 down tiles while preserving the original split-K reduction.

The accepted change selects GGML's existing **32×32 tiles for the feed-forward
gate/up and attention output projections** in live ViTPose. The combined
experiment reduces normalized inference from 62.06 to 54.77 ms (**11.75%**),
with identical output bytes. QKV and the feed-forward down projection keep
upstream's selection.

## Experiment

RTX 5070 Ti, strict F32 (F16 and both cooperative-matrix paths disabled), CPU
affinity 0–7, at most eight build jobs. A separate backend library reuses the
existing shader objects and changes only pipeline selection for the four
ViTPose projection shapes. The first sweep measures 60 inferences per variant
after ten warm-ups; the combined sweep measures 120, bracketed by baselines.
Each call processes a crop and flip, with 197 tokens each.

| Change from upstream 128×128 selection | Whole ViTPose ms | Exact heatmap bytes? |
| --- | ---: | :---: |
| Baseline, first sweep | 62.15 | Yes |
| Gate/up 64×64 | 58.32 | Yes |
| Gate/up 32×32 | 56.75 | Yes |
| Down 64×64 | 64.22 | No |
| Down 32×32 | 60.89 | No |
| QKV 64×64 | 61.72 | Yes |
| QKV 32×32 | 61.72 | Yes |
| Attention output 64×64 | 60.22 | Yes |
| Attention output 32×32 | 59.49 | Yes |
| All four 64×64 | 59.01 | No |
| Baseline, repeat | 61.92 | Yes |

The combined gate/up + attention-output 32×32 variant measures **54.77 ms**.
Adding QKV 64×64 gives 54.84 ms; adding QKV 32×32 gives 55.23 ms. Neither
improves the combination, so QKV remains unchanged. Combined-sweep baselines
measure 62.123 and 61.998 ms.

Down-projection variants alter split-K selection and the reduction order.
Their maximum heatmap difference is only 7.60e-7 on the synthetic input, but
they fail the stronger byte-identical requirement and were rejected.

## Interpretation

Small tiles give more independently scheduled workgroups and less unused
tile area for 197 token columns: seven 32-column tiles have capacity 224,
whereas two 128-column tiles have capacity 256. The smaller kernel also keeps
fewer output accumulators per thread. These source-level differences explain
why tile choice is worth testing, but the profile does not isolate their
individual effects or establish hardware occupancy/register-stall statistics.

The dot-product arithmetic remains the existing F32 shader implementation.
Both accepted shapes have reduction length 1280, below the backend's split-K
threshold; changing their output tiles preserves the reduction partition.
The final choice retains the down projection's original split-K behavior.

Final integrated GPU timestamps confirm where the gain occurs:

| Matmul group | Before | After | After throughput |
| --- | ---: | ---: | ---: |
| Gate/up, 64 calls | 24.993 ms | 20.173 ms | 16.38 TFLOP/s |
| Attention output, 32 calls | 5.080 ms | 2.581 ms | 16.00 TFLOP/s |
| Down, unchanged selection | 10.897 ms | 10.909 ms | 15.15 TFLOP/s |
| QKV, unchanged selection | 7.860 ms | 7.917 ms | 15.65 TFLOP/s |

Two final uninstrumented normalized benchmark runs average 54.897 and
54.942 ms. The timestamp-instrumented GPU span is 55.663 ms; timestamps add
barriers and synchronization, so it should not be subtracted from baseline
wall time to estimate host overhead.

**Power tradeoff:** sustained isolated ViTPose now averages about 300 W and
activates the software power cap, versus roughly 257 W previously. GPU activity
remains about 98.2%, SM clocks average 2.74 GHz and temperature is 63–64°C, with
no hardware thermal-slowdown flag. This is a latency improvement, not an
established energy-efficiency improvement. The complete worker alternates
ViTPose with other stages; its power consumption was not separately sampled.

## Final live replay and parity

The integrated backend is tested with the optimization enabled and with
`GEMX_VITPOSE_TILES=0`, each over the public 72-frame clip repeated three times.
There are 216 input frames, 215 complete poses and 187 full-window timed frames
per run. The enabled run's trace confirms activation of both intended shapes;
the disabled run emits no activation trace.

| Steady-state mean | Upstream selection | Tuned selection |
| --- | ---: | ---: |
| ViTPose including crop/decode | 62.476 ms | 55.065 ms |
| Detector | 55.307 ms | 55.391 ms |
| Complete native worker | 122.091 ms | 114.764 ms |
| Native worker throughput | 8.19 frames/s | 8.71 frames/s |

This is an **11.9% reduction in ViTPose time** and **6.0% in complete native
worker time**. These measurements exclude browser, camera and transport costs.
All 215 complete output poses are byte-identical to the accepted pre-tuning
baseline, including warm-up, window eviction and repeated-clip boundaries.

Separately, every heatmap for all 72 normalized real-video crops and their
flips matches the original backend byte-for-byte at both batch two (live) and
batch eight (offline). This checks heatmaps themselves rather than relying on
unchanged peak locations. The pinned ViTPose upstream heatmap/keypoint fixture,
the four existing CTest contracts, Python syntax and whitespace checks pass.

The optimization changes no model parameters, precision, flip augmentation,
input resolution, temporal window or decoder math. Byte identity is established
for these fixtures and replay inputs, not an exhaustive proof over all possible
floating-point inputs or future drivers.

## Integration and rollback

`GEMX_VITPOSE_TILES` defaults to ON in CMake. The adapter builds a generated
copy of `ggml-vulkan.cpp` with a small selection hook; the pinned GGML submodule
stays pristine. Configuration fails if the expected insertion point is absent
or ambiguous. The hook applies only when all of these match:

- Physical device name `NVIDIA GeForce RTX 5070 Ti`;
- F16 and cooperative-matrix paths disabled; both matrix inputs F32;
- `N=197`, `K=1280`, batch two, and `M=5120` or `M=1280`;
- The existing small F32 pipeline is supported.

Offline batch eight, other hardware, other shapes, and other precision paths
keep upstream selection. A one-frame remainder in an offline job has the same
batch-two shape and may use the validated selection.

Set **`GEMX_VITPOSE_TILES=0`** in the worker environment to disable it without
rebuilding. Configure with **`-DGEMX_VITPOSE_TILES=OFF`** to compile the original
upstream source. `GEMX_VITPOSE_TILES_TRACE=1` logs each selected shape once per
thread, allowing activation to be checked independently of timing.

The demo loads `build/vulkan/bin/libggml-vulkan.so` when it starts a native
worker. A new live session picks up the rebuilt backend; an already-running
worker must finish/restart before it uses the new library.

## Reproduction

The scripts never change the production backend during a sweep:

```sh
taskset -c 0-7 python scripts/experiment_vitpose_tiles.py \
  --output generated/my-vitpose-tiles --iterations 120
```

This needs an existing Release Vulkan build and its `compile_commands.json`.
The script compiles an isolated copy of the upstream source, reuses generated
shader objects, and writes timing CSVs, final heatmaps and a JSON comparison.
`--cases baseline up-proj-s baseline-repeat` selects a smaller experiment.
Experimental down-projection and all-medium cases deliberately record
non-identical outputs; they are not candidates for automatic deployment.

`scripts/check_vitpose_crops.py` compares every heatmap batch from normalized
real-video crops, including flips, between separately loaded backends. It
defaults to batches two and eight. `scripts/profile_live_worker.py --module`
can replay complete poses with an isolated backend. Raw experiment artifacts
are under `generated/vitpose-tiles-20260919/`.

See [machine-readable results](vitpose-tiles-2026-09-19.json) for the sweeps,
final operator timings, live replay timings and validation evidence.
