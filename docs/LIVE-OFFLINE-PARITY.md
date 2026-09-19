# Live and offline parity — 2026-09-18 follow-up

Both native inference paths now have direct upstream regression coverage.
Independent end-to-end results on the pinned 72-frame football clip are close,
but are **not bit-identical**. This supersedes the earlier
[video audit](VIDEO-PARITY-AUDIT.md). It is not a multi-video quality evaluation
or a robot/camera latency validation.

## Independent video results

Native and upstream each detect their own boxes and compute their own
observations from identical RGB pixels. The following results use strict F32
on both sides: native Vulkan F16/cooperative-matrix paths disabled, upstream
CUDA TF32 disabled, Body F32 for offline, and no Body for live. OpenCV is pinned
to upstream's 4.11.0.86. Upstream default CUDA arithmetic produces different
results; it must not be mixed into a strict numerical reference silently.

| Comparison | Mean joint distance | Maximum joint distance |
| --- | ---: | ---: |
| Offline, pelvis-relative camera joints, with contact correction | 0.877 mm | 4.628 mm |
| Offline, world joints including trajectory and grounding | 3.496 mm | 9.968 mm |
| Live, pelvis-relative gravity-aligned joints, 30-frame rolling window | 0.984 mm | 19.051 mm |

Offline evaluates all 72 frames. Live emits 71 newest-frame predictions after
the two-frame warm-up, including every window eviction. Live shape is taken
from each emitted prediction, matching upstream; it is not averaged over the
whole emitted stream. Distances cover all 77 SOMA joints, including fingers.

The remaining live outliers are traceable to tiny detector-box differences
changing a ViTPose heatmap peak: at input frame 49 the thumb endpoint moves
about 0.67 source pixels, and at frame 67 the wrist moves about 0.67 pixels.
Their influence persists through the rolling context. Mean live keypoint
distance is 0.00121 px; maximum is 1.126 px. Offline mean box-coordinate
difference is 0.00000923 px; residual Body-token mean absolute difference is
0.00464. These residuals are reported, not hidden by sharing observations.

Raw-head thresholds in `compare_e2e_parity.py` are deliberately unchanged.
Independent end-to-end outputs still exceed those tight shared-input
thresholds. Passing component parity does not establish bit-exact end-to-end
parity. See the [machine-readable measurements](live-offline-parity-2026-09-18.json)
for raw-head errors, p95 distances, hashes and reference configuration.

## Changes

- Replaced YOLOX's incorrect 5-bit affine-style resize with OpenCV's 11-bit
  separable resize arithmetic. Pixel hashes match upstream across five sizes.
- Removed hidden F16 rounding inside YOLOX convolution's im2col buffer. A
  backend F32 flag alone could not undo the old rounding.
- Matched upstream detector score/NMS defaults (0.5/0.45), pre-smoothing box
  clipping, and full-image fallback when no person is detected.
- Added `GEMSEQ02`, retaining original float32 centre/size boxes separately
  from Body's corner boxes. The Go demo and audit runner use this format;
  `GEMSEQ01` remains readable.
- Added `gemx_session_create_live`: absent image conditioning, present CLIFF
  conditioning and unclamped decoded body scale, exactly as the released live
  ONNX/webcam path. `body_features` may be NULL; no SAM3D model is needed.
- Added the checkpoint contact head, `gemx_infer_contacts`, and
  `gemx_refine_contacts`: translation correction, x/z smoothing, grounding and
  two-pass CCD IK. `gemx-pipeline --offline-contact` enables this path.
- The offline demo now defaults to strict F32, F32 Body and contact correction.
  `--strict=false --bf16 --contacts=false` explicitly selects the approximate
  path without contact refinement; it is not the parity configuration.

The earlier decoder-statistics and ViTPose channel-order fixes remain in place.

## What is tested

- Existing offline network fixtures at lengths 1, 2, 16, 30 and 120, plus motion,
  SOMA and GLB checks, pass on CPU and strict Vulkan.
- New live fixture executes all 71 rolling inference calls. With identical
  observations, maximum raw-motion error is `4.27e-6` on strict Vulkan and
  `2.87e-6` on CPU. Body pose, identity and scale are checked independently.
  Float32 `acos` near identity in upstream's yaw rollout amplifies rounding;
  its orientation-component difference is bounded separately at 0.004 rad
  (observed maximum 0.00227 rad).
- Contact refinement is checked against actual upstream before/after outputs:
  maximum translation error is `7.16e-7` m and body axis-angle error is
  `0.000684` rad. Nonfinite contacts must fail without modifying motion.
- Release/Vulkan CTest suites and Go tests pass, including resize hashes and
  the new sequence-manifest layout.

Fixtures are produced by `generate_live_reference.py` and
`generate_contact_reference.py` from independent upstream captures, rather
than duplicated native equations. Their SHA-256 identities and live source
identity are recorded in `reference/sources.json`.

## Reproduce

Convert the standard pinned ONNX model with its official checkpoint to include
both live conditioning and the contact head (PyTorch is conversion-only):

```sh
taskset -c 0-7 python scripts/convert_onnx_to_gguf.py \
  generated/reference/onnx/gem_denoiser.onnx \
  generated/reference/gem-x-contact-f32.gguf \
  --upstream-root /path/to/GEM-X \
  --soma-rig generated/reference/SOMA_neutral.npz \
  --identity-data generated/reference/soma-identity-native.npz \
  --checkpoint generated/reference/gem_soma.ckpt \
  --gguf-py /path/to/llama.cpp/gguf-py
```

Older 246-tensor GGUFs still support unrefined offline inference. Live mode
requires reconversion (247 tensors); contact inference requires `--checkpoint`
(251 tensors). The offline demo default points at the latter filename.

`capture_video_reference.py --strict-f32` independently captures both upstream
offline variants and exports only source RGB frames for native use. Run it in
the pinned reference environment described in the earlier audit. Then:

```sh
taskset -c 0-7 python scripts/run_video_native_audit.py \
  --frames /path/to/upstream-capture/frames --output /path/to/new-native-run
```

This uses an empty output directory, at most eight CPU cores, the same native
box/manifest arithmetic as the demo, and local models. It writes both
`f32/unrefined-output` and `f32/output` (with contacts). Compare the latter to
`hpe_results.pt` using `audit_video_parity.py`; compare unrefined output to
`hpe_results_nopost.pt`.

For live, `capture_live_reference.py` imports the actual upstream
`webcam_stream.py` and runs its `process_frame` unchanged, redirecting model
loading to explicit local assets. Use the pinned absent-image ONNX asset,
`--strict-f32` and `--window 30`. Then replay native observations independently:

```sh
taskset -c 0-7 env OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
  GGML_VK_DISABLE_F16=1 GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 \
  python scripts/replay_live_native.py \
  --library build/vulkan/libgemx.so --module build/vulkan/bin/libggml-vulkan.so \
  --gem generated/reference/gem-x-contact-f32.gguf \
  --vitpose generated/reference/vitpose-f32.gguf \
  --yolox generated/reference/yolox-f32.gguf \
  --frames /path/to/source/frames --output /path/to/native-live.npz --window 30
```

Use `audit_live_parity.py` for decoded joint comparison. Its `--help` documents
SOMA-source/assets arguments. The native replay harness also has an explicit
`--observations` diagnostic option; results from that option must be labelled
shared-input, never independent end-to-end evidence.

The optional reference executables require local models:

```sh
taskset -c 0-7 build/release/gemx-live-reference-test \
  generated/reference/gem-x-contact-f32.gguf build/release/bin reference/gem_live_reference.bin
taskset -c 0-7 build/release/gemx-contact-reference-test \
  generated/reference/gem-x-contact-f32.gguf build/release/bin reference/gem_contact_reference.bin
```

All local captures are under `generated/reference/e2e/audit-20260918/`.
Final offline results are `manifest2/f32/output`, `offline-final-audit.json`
and `offline-final-skeletons.png`; live results are `live-native-full-f32.npz`
and `live-audit.json`. The live harness measured roughly 134 ms per accepted
frame including graph warm-up, excluding model loading/camera/network. This
is replay timing, not measured camera-to-robot latency.

## Remaining scope

The native live inference API and replay path are implemented. The subsequent
demo update also connects continuous browser capture to a resident native
worker; lossless HTTP replay matches the accepted native results exactly over
all 71 emitted poses. Browser integration is checked with a simulated webcam.
SONIC publishing remains unimplemented and no robot was connected. Live omits
Body entirely, as requested.
See [the live integration design](LIVE-CAMERA-DESIGN.md) for remaining wiring.
Sequences beyond 120 frames still use the native consecutive-window policy;
parity with upstream's long-sequence local attention is not established.
