---
license: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
base_model: nvidia/GEM-X
library_name: gguf
inference: false
tags:
  - gguf
  - ggml
  - human-motion
  - pose-estimation
  - gem-x
  - vulkan
---

# GEM-X SOMA — GGUF conversion for gem-x.cpp

F32 format conversions of NVIDIA GEM-X and its observation components for
**gem-x.cpp**, a C++23/GGML CPU/Vulkan runtime. This is an independent port,
not an NVIDIA release, fine-tune, retraining, or low-bit quantization. These
custom architectures require gem-x.cpp; they are not LLM GGUFs for llama.cpp.
This card and bundle manifest are prepared locally. No model upload or new
public download location is implied.

## Files and compatibility

| File | Contents | Bytes |
| --- | --- | ---: |
| `gem-x-contact-f32.gguf` | GEM regression model, absent-image conditioning, contact head, SOMA skeleton/identity data | 177,334,400 |
| `vitpose-f32.gguf` | GEM-X's DINOv3 ViT-H observation network; 77 heatmaps | 3,387,861,248 |
| `yolox-f32.gguf` | YOLOX-X HumanArt detector | 396,106,720 |

SHA-256 identities of the tested artifacts are in [manifest.json](manifest.json).
Floating weights are F32; structural arrays may be integer tensors. These are
format/graph conversions rather than bit-for-bit copies of the upstream archive.
GEM-X's matrices are transposed/remapped to GGML layouts, ONNX initializers are
selected for the inference graph, and verified rig/identity constants are
included. The contact variant has 251 tensors; older 246-tensor conversions
lack live absent-image conditioning and contact weights. Reconvert older files
for the current default live/offline demo.

Live inference needs these three files. Offline inference additionally needs
sam3d.cpp's compatible Body backbone, pose branch and MHR GGUFs, with
`sam3d-body-infer --gem-features` producing the 1024-value pose token. Those
separately licensed assets are not included in this bundle. Loading all networks,
workspaces and graph buffers requires more RAM/VRAM than the file sizes alone.

## Model and interfaces

The temporal network has 12 blocks, a 512-wide embedding, eight attention heads
and a 2048-wide feed-forward expansion. It consumes per-frame SOMA-77 image
keypoints/confidences, a person box, camera intrinsics, camera angular motion,
and (offline only) a compatible Body feature vector. The model selects its
published observation subset internally. Native outputs include 585-value raw
motion vectors, weak-perspective camera parameters, 76 local body rotations,
45 identity coefficients, 69 scale parameters, root orientation/translation,
and a 77-joint skeleton. Exports are skeleton-only animated GLB, not a body mesh.
See [gemx.h](../include/gemx.h) for exact array shapes and ownership.

Live mode follows upstream's absent-image configuration: a rolling 30-frame
window, two-frame warm-up and newest-frame output. It does not use SAM3D Body,
offline contact refinement or offline world-trajectory reconstruction. Detection
interval 1 matches upstream cadence; the demo starts at 5 and lets the user
choose. Crop reuse at larger intervals changes the observation sequence.

Offline regression fixtures cover up to 120 frames. The API accepts longer
sequences up to 4096 with consecutive windows; that policy is not equivalent
to upstream overlapping local attention beyond 120 frames. The demo limits
completed clips to 120 sampled frames.

## Provenance and conversion

- GEM-X source: `NVlabs/GEM-X` revision
  `32992550dba114c62243fb55e361311972dce8f9`.
- Model source: `nvidia/GEM-X` revision
  `5ccf5ca3746c3620aa4016114f069a5f6ae399cd`.
- SOMA source: `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`;
  neutral rig from `nvidia/SOMA-X` revision `466879a`.
- Detector: MMPose's YOLOX-X HumanArt `a39d44ed` ONNX archive.

[reference/sources.json](../reference/sources.json) pins source URLs, byte sizes
and SHA-256 identities, including checkpoint, ONNX external data and fixtures.
Converters reject unexpected official artifacts. Python dependencies are
conversion-only: NumPy, ONNX, `gguf`, and PyTorch for verified checkpoint/contact
or MHR/SOMA identity extraction. Use a separate conversion environment.

```sh
python3 scripts/download_reference.py
python3 scripts/download_yolox.py
# Obtain the hash-pinned official checkpoint, SOMA rig and MHR/SOMA assets
# listed in reference/sources.json. Extract identity constants with:
python3 scripts/extract_soma_identity.py /path/to/verified/assets \
  generated/reference/soma-identity-native.npz --soma-source /path/to/SOMA
python3 scripts/convert_onnx_to_gguf.py \
  generated/reference/onnx/gem_denoiser.onnx \
  generated/reference/gem-x-contact-f32.gguf \
  --upstream-root /path/to/GEM-X \
  --soma-rig /path/to/SOMA_neutral.npz \
  --identity-data generated/reference/soma-identity-native.npz \
  --checkpoint /path/to/gem_soma.ckpt
python3 scripts/convert_vitpose_onnx_to_gguf.py \
  generated/reference/onnx/vitpose.onnx generated/reference/vitpose-f32.gguf
python3 scripts/convert_yolox_onnx_to_gguf.py \
  generated/reference/yolox-humanart.onnx generated/reference/yolox-f32.gguf
python3 distribution/verify.py --models generated/reference
```

The examples use an installed `gguf` Python package; `--gguf-py` can select a
specific package checkout instead. Model files are ignored by source control.
The manifest describes the actual tested files; a converter/toolchain change
may require a newly validated manifest rather than assuming matching bytes.

## Usage and evaluation

Build the native runtime as described in the [README](../README.md), place the
files at the configured model paths and launch the [demo](../demo/README.md).
Runtime inference requires neither Python nor CUDA. CPU and Vulkan are supported;
Vulkan tile optimizations are guarded to validated hardware, with generic
fallback selection elsewhere. Strict F32 is the demo default; approximate
BF16/F16 modes should not be presented as the same parity configuration.

On the pinned annotated 72-frame football clip, independent upstream/native
strict-F32 comparisons yielded mean pelvis-relative joint differences of
**0.88 mm offline** and **0.98 mm live**, with a **19.05 mm worst live joint**.
These measure agreement with upstream, not ground-truth pose accuracy. They
are not evidence of general real-world accuracy or bit-exact upstream parity.
See [the parity report](../docs/LIVE-OFFLINE-PARITY.md).

Recorded RTX 5070 Ti performance: about **48.92 ms** for strict-F32 ViTPose
including flip augmentation, and **15.35 fps** for the simulated-camera live
pipeline with detection interval 5. These differ in scope and exclude physical
camera latency; see [live measurements](../docs/LIVE-PIPELINING.md).

No training was performed for this conversion. Training/data claims belong to
the [upstream model card](https://huggingface.co/nvidia/GEM-X), not an independent
assessment by this project. The conversion was evaluated on component fixtures
and the pinned clip, not a representative demographic or clinical benchmark.

## Intended use and limitations

Research, animation, motion capture and integration experiments. Monocular scale,
occlusion, unusual poses, moving cameras and crop errors can degrade estimates.
Crop reuse assumes the person stays near the previous detection. Hand detail and
metric depth may be unreliable. This demo does not publish commands to SONIC or
control a G1, and has not been validated for autonomous physical robot control,
medical decisions or other safety-critical applications. Obtain permission for
input footage and consider the sensitivity of movement recordings.

## Licenses and attribution

The primary NVIDIA model retains the
[NVIDIA Open Model License](../LICENSES/NVIDIA-Open-Model-2025-10-24.pdf), separate
from the port's Apache-2.0 source license. Preserve [NOTICE](NOTICE) with a model
bundle. ViTPose's DINOv3 component retains the
[DINOv3 license](../LICENSES/DINOv3.md); embedded MHR data retains its
[MHR](../LICENSES/MHR-Apache-2.0.txt) and applicable
[Momentum](../LICENSES/Momentum.txt) notices. Detector components retain their
[YOLOX](../LICENSES/YOLOX-Apache-2.0.txt) and upstream MMPose notices. SAM3D Body
is a separate dependency under Meta's terms. This card's primary license field
does not replace component-specific terms.

Credit for the models and research belongs to NVIDIA, Meta and the detector
and tracking authors. See [source licensing/provenance](../docs/LICENSING.md).
