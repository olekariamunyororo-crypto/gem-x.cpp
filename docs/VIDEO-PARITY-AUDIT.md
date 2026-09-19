# Video parity audit — 2026-09-18

> Historical intermediate audit. Superseded by [the live/offline follow-up](LIVE-OFFLINE-PARITY.md), which fixes the detector discrepancies and implements contact refinement. Measurements below describe the earlier state.

**Independent video parity is not yet achieved.** Two conversion/preprocessing
bugs were fixed, but the fresh native pipeline still differs materially from
upstream. Passing component fixtures had overstated the previous coverage.
The live-camera investigation is in [LIVE-CAMERA-DESIGN.md](LIVE-CAMERA-DESIGN.md).

## Scope and reference

The test uses all 72 frames of the official annotated football GIF, losslessly
decoded to 480×270 RGB at 10 FPS. This is one clip, not a multi-video quality
evaluation, and the published mesh renders used a clean source we do not have.
Numerical agreement with upstream is not ground-truth motion accuracy.

Both pipelines independently ran detection/tracking, ViTPose, Body feature
extraction and GEM. The decoded RGB SHA-256 matches between them:
`c940f48263ce9f16e08f409868a4f944158edab999b4d47c1f90d9aebfeafef0`.
Upstream GEM revision is `32992550dba114c62243fb55e361311972dce8f9`,
with SOMA revision `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`.
The reference uses OpenCV 4.11.0.86, ONNX Runtime GPU 1.23.2 and Torch
2.7.1+cu128, with batch-one Body and ViTPose extraction. The existing container
had OpenCV 5; its intermediate audit results are excluded from these findings.
Native results use Vulkan on an RTX 5070 Ti and separately recomputed BF16/F32
Body tokens. F32 here describes Body; downstream Vulkan fast paths remain on.

The primary comparison uses upstream `postproc=False`. Its optional contact
correction, IK and grounding are not implemented natively. Full parity with
the default postprocessed demo therefore remains a separate missing feature.
Skeleton distances use all 77 joints and upstream SOMA forward kinematics,
with the same temporal-mean identity/scale policy as the native exporter.

## Bugs fixed

1. **Motion statistics were decoded incorrectly.** The released SOMA-v2 config
   sets `clip_std: true`, meaning standard deviations are clamped to at least
   one. Conversion and the old copied-equation fixture omitted this. Native
   loading now clamps statistics for existing GGUF files, and conversion
   writes the correct values. The reference generator now calls the actual
   configured upstream decoder and motion transforms.
2. **ViTPose received the wrong channel order.** Upstream reverses channels
   inside its crop preprocessing. Native previously preserved RGB. Native now
   reproduces that reversal and OpenCV 4 affine rounding; fixtures were
   regenerated through the actual upstream helpers and official ONNX model.

An isolated decoder test using the older shared-input capture reduced mean
pelvis-relative camera-space joint disagreement from **236.4 mm to 1.56 mm**.
This demonstrates the decoder bug's impact, but is not independent end-to-end
evidence. Rebuild and reprocess old outputs; old cached skeletons retain the bug.

## Independent results after fixes

| Metric | BF16 Body | F32 Body |
| --- | ---: | ---: |
| Pelvis-relative camera joints, mean | 24.33 mm | 24.05 mm |
| Pelvis-relative camera joints, p95 | 71.24 mm | 71.03 mm |
| Pelvis-relative camera joints, maximum | 214.14 mm | 209.14 mm |
| Camera joints including translation, mean | 50.43 mm | 49.44 mm |
| World joints including rollout, mean | 144.37 mm | 146.91 mm |
| Raw motion head, mean absolute error | 0.006370 | 0.006356 |
| Raw camera head, mean absolute error | 0.006625 | 0.006592 |
| Body token, mean absolute error | 0.01646 | 0.01706 |

Both runs use the same native detector and ViTPose results. Smoothed box
coordinates differ from upstream by 0.0428 px mean / 0.1557 px maximum.
ViTPose positions differ by 1.638 px mean / 87.50 px maximum; the confidence
mean absolute error is 0.01043. The keypoint metric includes low-confidence
joints. F32 Body alone does not resolve these discrepancies.

The durable [machine-readable report](video-parity-2026-09-18.json) includes
artifact hashes and additional metrics. Local artifacts are under
`generated/reference/e2e/audit-20260918/`: `final-audit.json`,
`final-skeletons.png`, and `fresh-native/{bf16,f32}/final-output/motion.glb`.
`decoder-before-after.png` illustrates the isolated decoder correction.

The older upstream capture reused native boxes and keypoints (the saved
keypoints are byte-identical). Its tight network errors cannot establish
independent observation-stage or end-to-end parity. Previous documentation
attributing the entire remaining gap to BF16 was incorrect.

## Validation and remaining work

Release and Vulkan CTest suites pass (three tests each), as do the Go demo
tests. GEM network tests pass for lengths 1/2/16/30/120, and regenerated
30-frame decoder/SOMA fixtures pass on CPU and Vulkan. Maximum skeleton
coordinate error is about 0.015 mm on
CPU and 0.570 mm on Vulkan. The independent ViTPose fixture has maximum CPU
heatmap error `5.22e-7` and keypoint-component error `1.79e-7`; Vulkan heatmap
error is `0.001263`, with four joint peaks displaced over 1 px, at most 2.04 px.
These component results do not override the video discrepancies above.

Next, hold upstream boxes fixed and compare each observation stage on these
real frames, then hold upstream observations fixed and isolate GEM. This will
separate crop sensitivity, reduced-precision peak selection and Body feature
differences. Native detector defaults also differ (score/NMS 0.1/0.65 versus
upstream 0.5/0.45), and clipping around smoothing needs alignment. Small box
error on this clip does not establish equivalent tracking on crowded videos.
Then add clean videos with turns, occlusions and fast movement, and evaluate
the contact/grounding gap separately. The residual error is not yet diagnosed.

## Repeating the audit

`scripts/capture_video_reference.py --help` lists the local checkpoint, source
and video arguments. Run it in the pinned upstream environment with GEM-X,
SOMA and SAM3D on `PYTHONPATH`, DINOv3 available locally, and `inputs/soma_assets`
and `inputs/soma_data` available in its working directory. It produces fresh
boxes, keypoints, tokens, both postprocessing variants and `capture.json`;
it does not consume native intermediates.

Generate fresh native offline outputs from the same RGB frames, separately
for each Body mode. `scripts/audit_video_parity.py --help` describes the
upstream-result, SOMA-assets and repeated `--native LABEL=DIR` arguments;
`--plot` creates the skeleton comparison. `scripts/compare_e2e_parity.py`
provides raw-head acceptance thresholds; the audit script reports distances
without declaring them passing. Reference scripts require trusted local assets.

All compilation and inference in this audit were restricted to CPU cores 0–7,
with at most eight build jobs and explicit eight-thread inference limits.
Kimodo processes were stopped to release the GPU, as authorized.
