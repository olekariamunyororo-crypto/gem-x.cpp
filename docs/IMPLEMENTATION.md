# Native GEM-X implementation record

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
7. The public APIs support YOLOX detection selected by an initial manual box,
   ByteTrack identity association, complete offline sequences, fixed-context
   live operation, and skeleton animation export. They form the
   inference adapter for the existing `sam3d.cpp` photo/video/live demo, whose
   selection, bounded frame pipeline, recording, and export controls are
   already shared across body paths.
8. The measured obvious optimizations are implemented: batch 1..8 ViTPose
   graphs, original/flip fusion, parallel RGB crop preparation for up to four
   source frames, bounded graph caches, backend-resident live ring inputs,
   per-frame partial uploads, fixed position uploads, logical ring reordering
   on device, fixed graph reuse, and newest-only prediction downloads. Offline
   inference still returns every requested frame.

## Numerical policy

CPU learned outputs are compared closely enough to detect changed arithmetic,
layout, preprocessing, or tensor mapping. The default Vulkan fast path permits
accumulated reduced-precision error while checking final decoded motion and the
SOMA skeleton. Current denoiser maxima against upstream are `0.00118` raw and
`0.000608 m` at the skeleton. ViTPose heatmaps differ by at most `0.00136`.

Heatmap argmax is discontinuous: a small reduced-precision heatmap change can
select a neighboring peak. In the reference image this affects four of 77
joints by more than one pixel, all within 6.45 px, with the worst at the
low-confidence `RightHandIndexEnd`. This divergence is recorded rather than
hidden by a looser heatmap comparison. Disabling Vulkan F16 and cooperative
matrix paths yields maximum heatmap and keypoint errors below `1e-6`.

## Parity scope

| Boundary | Evidence | Status |
|---|---|---|
| GEM raw heads, identical input tensors, L=1/2/16/30/120 | Official ONNX fixture vs native CPU/Vulkan | Covered |
| Published motion postprocess and SOMA forward kinematics | Official Python fixture vs native CPU/Vulkan | Covered |
| ViTPose and YOLOX, fixed input images | Official ONNX outputs vs native CPU/Vulkan | Covered, with documented reduced-precision peak movement |
| SAM 3D Body 1024-value token, standard F32 path | Captured upstream CUDA tensor vs native Vulkan tensor | Covered on the captured exemplar |
| Default demo SAM 3D Body token | BF16 encoder vs upstream/captured F32 token | Approximate |
| Complete real video through detection, Body, ViTPose, GEM and rendering | No paired upstream/native video exemplar yet | Not established |
| Live output | Causal prefix adaptation of an offline bidirectional network | Not upstream-equivalent |
| Sequences longer than 120 frames | Consecutive native windows vs upstream local attention | Not established at boundaries |

On the current Body exemplar, the standard BF16 pose token differs from the
standard F32 token by `0.00457` maximum and `0.000897` mean absolute error. The
discarded `fast384` experiment differs by `0.208` maximum and `0.0488` mean and
is not used by the demo.

The live ring is tested against upstream-length one- and two-frame prefixes.
Unfilled keys are attention-masked, and identity/scale reductions include only
real frames. Its history tensors use a separate backend buffer because a normal
graph allocator may reuse dead input storage between computations.

This does not make live operation identical to upstream video inference. The
released network is bidirectional: an offline result for frame N uses later
frames, while the live adapter can only use the prefix available at that time.
The live adapter preserves its inferred root translation across pushes, using
the previous prefix's velocity. Upstream may revise that velocity once future
frames arrive. Raw-model fixture parity applies to complete tensor sequences of
at most 120 frames and to each live prefix's newest raw prediction, not to a
completed offline clip reconstructed from previously emitted live samples.

Sequences longer than 120 frames are currently split into consecutive 120-frame
windows. The released Python path instead applies a local attention window over
the complete sequence, so the boundary frames are not yet parity-covered.

The interactive adapter follows upstream's 192:256, 1.2x detector-box
conversion for both ViTPose and SAM 3D Body, and uses upstream's
`max(width,height)` default intrinsics for GEM conditions. Live identity choice
and gap handling are necessarily causal; upstream chooses the dominant track,
interpolates gaps, and smooths boxes after reading the complete clip.

## Performance record

Representative release measurements on an AMD Ryzen 9 7900 and RTX 5070 Ti:

| Component | Configuration | Time / throughput |
|---|---|---:|
| GEM denoiser | CPU, 2 cores, L=120 | 113.5 ms / 1057 output frames/s |
| GEM live | CPU, 2 cores, context 120 | 115.4 ms / 8.66 pushes/s |
| GEM denoiser | Vulkan, L=120 | 3.38 ms / 35,519 output frames/s |
| GEM live | Vulkan, context 120 | 3.26 ms / 307 pushes/s |
| ViTPose + flip | Vulkan, batch 2 | 33.1 ms / 30.2 source frames/s |
| ViTPose + flip | Vulkan, batch 8 | 86.0 ms / 46.5 source frames/s |
| YOLOX-X HumanArt | Vulkan, warm integrated frame | 27.5 ms |

The live optimization reduces host preprocessing/upload/download work from
about 0.158 ms to 0.034 ms per Vulkan push. Attention still recomputes the full
bidirectional 120-frame context, which preserves the released model semantics.
The main end-to-end throughput limit is now the Body and ViTPose observation
stage, so applications should overlap those independent tasks and batch short
offline runs. Compilation is always limited to eight jobs.
