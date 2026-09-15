# gem-x.cpp

Native C++23/GGML inference for [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X),
targeting CPU and Vulkan. The library consumes SOMA-77 image observations,
camera data, and the compatible 1024-value pose token exposed by `sam3d.cpp`.
It produces temporally coherent SOMA-77 motion and skeleton-only animated GLB.

The released regression components are implemented natively:

- exact GEM-X preprocessing, twelve temporal blocks, output heads, and motion
  postprocessing;
- the official DINOv3 ViT-H ViTPose-77 observation model, including OpenCV
  compatible crops, flip augmentation, and UDP peak refinement;
- the YOLOX-X HumanArt person detector used by the official demo, with
  ByteTrack identity association and causal three-frame box smoothing;
- learned SOMA identity fitting through the MHR-to-SOMA transfer data;
- complete offline outputs up to 4096 frames, with exact upstream parity tested
  through 120 frames and consecutive-window handling beyond that;
- a fixed-context live API with backend-resident ring inputs, fixed graph reuse,
  a 588-float newest-frame download, and causal root-translation state.

`sam3d.cpp` must run its Body pose branch in GEM-X feature mode when producing
the 1024-value token. The `sam3d-body-infer --gem-features` option selects that
mode. ViTPose and Body image preparation are independent and can run
concurrently; the GEM denoiser starts after both observations are available.
The `gemx-pipeline` executable implements the existing `sam3d.cpp` demo's
persistent worker protocol. A user's initial box selects an identity; each
subsequent frame is detected and tracked before both ViTPose and SAM 3D Body
consume the resulting box. It supports independent stream resets, native
`GEMPOSE1` recording samples, and animated GLB export from a bounded manifest.

Parity is established at component boundaries with pinned upstream fixtures.
It has not yet been established for one complete real video run and its final
render. The demo is also a causal adaptation of an offline bidirectional model,
and its default SAM 3D Body encoder uses BF16. Those two paths should not be
described as numerically identical to upstream. The precise coverage and
remaining gaps are recorded in `docs/IMPLEMENTATION.md`.

## Build

```sh
cmake --preset debug
cmake --build --preset debug -j8
ctest --preset debug
```

The build sets `GGML_CPU_ALL_VARIANTS=ON` and `GGML_NATIVE=OFF`. It emits
portable CPU variants, including AVX2 and AVX-512 variants where supported by
the compiler, and GGML selects the best compatible module at runtime. Vulkan is
available through the `vulkan` preset. Keep compilation at eight jobs or fewer.

Runtime inference uses GGUF files produced from hash-verified official assets.
PyTorch, ONNX Runtime, OpenCV, and Python are conversion/reference dependencies;
the installed runtime library does not depend on them. Asset identities and
hashes are pinned in `reference/sources.json`; generated weights remain ignored.
The detector can be prepared without broad archive extraction:

```sh
python scripts/download_yolox.py
python scripts/convert_yolox_onnx_to_gguf.py \
  generated/reference/yolox-humanart.onnx \
  generated/reference/yolox-f32.gguf --gguf-py /path/to/gguf-py
```

## Validation and performance

The checked reference fixtures cover preprocessing, denoiser lengths 1, 2, 16,
30, and 120, decoded motion, SOMA/MHR skeletons, and ViTPose heatmaps/keypoints.
Against the official ONNX Runtime CPU result, native CPU ViTPose has a
`6.26e-7` maximum heatmap error and `2.69e-7` maximum keypoint-component error.
The Vulkan fast path has a `1.36e-3` maximum heatmap error. Four of 77 peak
locations move by more than one pixel, with a 6.44 px worst case at a low
confidence hand endpoint; strict Vulkan F32 mode restores sub-micro-unit parity.
Native YOLOX differs from the official ONNX Runtime person box by at most
0.080 px on CPU and 0.065 px on Vulkan. Warm Vulkan detection takes about 27 ms
on an RTX 5070 Ti in the integrated demo.

On an RTX 5070 Ti, the default Vulkan GEM denoiser evaluates a 120-frame window
in about 3.38 ms. The resident live path takes about 3.26 ms per push (307 Hz),
while uploading one observation and downloading one prediction. ViTPose with
flip augmentation takes about 33.1 ms per source frame at batch 2; batching four
source frames plus their flips reaches about 46.5 source frames/s. These figures
describe the learned components independently and do not include SAM 3D Body,
image decode, or application scheduling.

See `docs/IMPLEMENTATION.md` for parity thresholds, known numerical divergence,
the causal live/offline distinction, and the step-8 optimization record.
