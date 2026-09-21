# Native GEM-X implementation record

Historical measurements and implementation notes. For current defaults and transport, see
[VITPOSE-DEFAULTS.md](VITPOSE-DEFAULTS.md) and [LIVE-PIPELINING.md](LIVE-PIPELINING.md).

The behavioral reference is NVIDIA GEM-X revision
`32992550dba114c62243fb55e361311972dce8f9`. SOMA is pinned at
`e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`. Official asset byte counts and
SHA-256 values are recorded in `reference/sources.json` and checked before any
ONNX or checkpoint payload is interpreted.

## Released contract

- Inputs are keypoints `[L,77,3]`, boxes `[L,3]`, intrinsics `[L,3,3]`, Body
  pose tokens `[L,1024]`, and camera angular velocity `[L,6]`.
- The released model selects 33 observed joints, embeds conditions to width 512,
  and runs twelve eight-head RoPE encoder blocks.
- Raw output is 585 values per frame: 456 local rotation-6D values, 45 identity
  coefficients, 69 scale values, two 6D global rotations, and 3 local velocity
  values. The camera head adds three weak-perspective values.
- ViTPose accepts a 256x192 ImageNet-normalized crop and produces 77x64x48
  heatmaps. Native RGB inference combines original and horizontal-flip samples
  in one bounded graph.
- YOLOX-X HumanArt accepts a top-left-padded 640x640 BGR image. Native inference
  decodes the three person-class heads, applies NMS, and feeds ByteTrack before
  the published 1.2x ViTPose crop expansion.

## Completed gates

1. Official sources and model files are pinned and hash checked before safe
   tensor-only conversion to GGUF.
2. Keypoint normalization and BEDLAM/CLIFF camera conditions match upstream.
3. Every temporal block and raw head matches the official graph at lengths 1,
   2, 16, 30, and 120.
4. Motion denormalization, scale reconstruction, rotation decoding, camera
   conversion, and global trajectory rollout match the released Python path.
5. MHR-to-SOMA identity fitting, SOMA-77 forward kinematics, and animated GLB
   export match the upstream skeleton exemplar. CPU skeleton position error is
   at most 0.024 mm; default Vulkan is at most 0.608 mm.
6. `sam3d.cpp` exposes the compatible Body token mode, and native ViTPose
   reproduces the official ONNX model. CPU heatmap/keypoint maxima are
   `6.26e-7` and `2.69e-7` respectively.
7. The public APIs support YOLOX detection, ByteTrack identity association,
   complete offline sequences, and skeleton animation export. The standalone
   `gem-x.cpp` demo records or uploads a
   complete clip, applies upstream-style dominant-track selection, gap filling,
   and symmetric smoothing. Its result view overlays the camera-space 3D
   skeleton and ViTPose observations on the synchronized source frame, retains
   a global-motion view, and exports the animated skeleton. GEM-X is no longer
   exposed inside the `sam3d.cpp` demo.
8. The measured obvious optimizations are implemented: batch 1..8 ViTPose
   graphs, original/flip fusion, parallel RGB crop preparation for up to four
   source frames, bounded graph caches, and batched offline inference that
   returns every requested frame.

## Numerical policy

CPU learned outputs are compared closely enough to detect changed arithmetic,
layout, preprocessing, or tensor mapping. The default Vulkan fast path permits
accumulated reduced-precision error while checking final decoded motion and the
SOMA skeleton. Current denoiser maxima against upstream are `0.00118` raw and
`0.000570 m` at the skeleton. ViTPose heatmaps differ by at most `0.001263`.

Heatmap argmax is discontinuous: a small reduced-precision heatmap change can
select a neighboring peak. In the reference image this affects four of 77
joints by more than one pixel, all within 2.04 px, with the worst at the
low-confidence `RightHandIndexEnd`. This divergence is recorded rather than
hidden by a looser heatmap comparison. Disabling Vulkan F16 and cooperative
matrix paths yields maximum heatmap and keypoint errors below `1e-6`.

## Parity scope

| Boundary | Evidence | Status |
|---|---|---|
| GEM raw heads, identical input tensors, L=1/2/16/30/120 | Official ONNX fixture vs native CPU/Vulkan | Covered |
| Motion decoding and SOMA forward kinematics | Actual configured upstream decoder vs native CPU/Vulkan | Covered; contact/IK/grounding now covered by actual upstream fixture |
| ViTPose and YOLOX, fixed input images | Official ONNX outputs vs native CPU/Vulkan | Covered, with documented reduced-precision peak movement |
| SAM 3D Body 1024-value token, standard F32 path | Captured upstream CUDA tensor vs native Vulkan tensor | Covered on the captured exemplar |
| Default demo SAM 3D Body token | Strict F32 encoder | Independent video residuals documented below |
| Independent 72-frame video pipeline | Fresh upstream/native detections, observations and decoded skeletons | Strict F32 mean pelvis-relative error 0.877 mm offline, 0.984 mm live; see current audit |
| Sequences longer than 120 frames | Consecutive native windows vs upstream local attention | Not established at boundaries |

On the current Body exemplar, the standard BF16 pose token differs from the
standard F32 token by `0.00457` maximum and `0.000897` mean absolute error. The
discarded `fast384` experiment differs by `0.208` maximum and `0.0488` mean and
is not used by the demo.

Sequences longer than 120 frames are currently split into consecutive 120-frame
windows. The released Python path instead applies a local attention window over
the complete sequence, so the boundary frames are not yet parity-covered.

The offline demo follows upstream's 192:256, 1.2x detector-box conversion for
both ViTPose and SAM 3D Body, uses upstream's `max(width,height)` default
intrinsics for GEM conditions, chooses the dominant track, interpolates gaps,
and smooths boxes after reading the complete clip.

The [live/offline parity report](LIVE-OFFLINE-PARITY.md) records the current
independent comparison, numerical limits and reproduction commands. The old
`input_exact` reference reused native boxes/keypoints and is only a shared-input
diagnostic. Current independent results include native detector, ViTPose and
Body observations, with contact correction enabled offline and Body omitted live.

The audit corrected decoder statistics, ViTPose channel order, YOLOX resize
and hidden F16 im2col rounding, and preserved canonical float32 observation
boxes. Upstream OpenCV 4.11.0.86 is required for the reference crop semantics.
Contact correction, IK and grounding are now implemented. The live API follows
the absent-image ONNX's always-present CLIFF condition and unclamped scale.
Browser live capture now uses a resident native worker with a 30-frame
rolling window. SONIC publishing remains separate integration work.

An earlier shared-input test exposed a released-export bug that component
fixtures did not cover. The GEM-X checkpoint masks its CLIFF box-camera
condition when fewer than four 2D joints are confident. NVIDIA's published
ONNX hard-codes the condition as present. Native inference now preserves and
applies the checkpoint's presence projection. Historical shared-input Vulkan
errors after this correction were `0.0010164` maximum / `0.00004492` mean for
motion, and `0.0001867` maximum / `0.00004650` mean for camera. These are not
independent end-to-end results. Batch-one upstream Body extraction is retained
for reproducibility; its earlier batch-16 comparison showed materially
different tokens.

## Performance record

Representative release measurements on an AMD Ryzen 9 7900 and RTX 5070 Ti:

| Component | Configuration | Time / throughput |
|---|---|---:|
| GEM denoiser | CPU, 2 cores, L=120 | 113.5 ms / 1057 output frames/s |
| GEM denoiser | Vulkan, L=120 | 3.38 ms / 35,519 output frames/s |
| ViTPose + flip | Vulkan, batch 2 | 33.1 ms / 30.2 source frames/s |
| ViTPose + flip | Vulkan, batch 8 | 86.0 ms / 46.5 source frames/s |
| YOLOX-X HumanArt | Vulkan, warm integrated frame | 27.5 ms |

The main end-to-end throughput limit is the Body and ViTPose observation stage,
so applications should batch short offline runs. Compilation is always limited
to eight jobs.

The [2026-09-19 live profile](LIVE-PROFILING.md) measures the current strict
demo path. Exact resize coefficient reuse, contiguous Focus writes, direct
single-frame crop preparation and a bounded exact shape cache reduce native
frame time by 2.82%, with unchanged pose bytes in replay.
