# gem-x.cpp

Native C++23/GGML inference for [NVIDIA GEM-X](https://github.com/NVlabs/GEM-X),
targeting CPU and Vulkan. The library consumes SOMA-77 image observations,
camera data, and, for offline inference, the compatible 1024-value pose token
exposed by `sam3d.cpp`. Live inference omits Body features.
It produces temporally coherent SOMA-77 motion and skeleton-only animated GLB.

The released regression components are implemented natively:

- GEM-X preprocessing, twelve temporal blocks, output heads, and motion
  decoding, contact correction, grounding and IK;
- the official DINOv3 ViT-H ViTPose-77 observation model, including OpenCV
  compatible crops, flip augmentation, and UDP peak refinement;
- the YOLOX-X HumanArt person detector used by the official demo, with
  ByteTrack identity association, gap filling, and symmetric five-frame box
  smoothing for completed clips;
- learned SOMA identity fitting through the MHR-to-SOMA transfer data;
- complete offline outputs up to 4096 frames, with component fixtures through
  120 frames and consecutive-window handling beyond that.

`sam3d.cpp` must run its Body pose branch in GEM-X feature mode when producing
the 1024-value token. The `sam3d-body-infer --gem-features` option selects that
mode. The standalone `demo/` application accepts an uploaded video or records
a webcam clip in the browser, then processes the complete clip. It runs
YOLOX/ByteTrack first, extracts Body tokens with a resident SAM3D worker,
batches ViTPose, evaluates GEM-X once over the full sequence, previews the
SOMA-77 skeleton, overlays its camera projection against the ViTPose evidence,
and exports an animated GLB. Offline clips are bounded to the component-tested 120-frame context.
The **Live webcam** mode continuously captures frames into a resident native
worker, runs upstream-style 30-frame rolling inference without Body features,
and displays the newest skeleton and synchronized camera overlay.

Both offline and rolling live inference now have upstream regression coverage.
On the pinned 72-frame clip, strict F32 independent runs achieve mean
pelvis-relative joint distances of **0.88 mm offline** and **0.98 mm live**.
They are not bit-identical: the worst live joint differs by 19.05 mm.
See [the current parity report](docs/LIVE-OFFLINE-PARITY.md) for measurements,
reproduction and limitations. The demo now connects live camera capture to that native path.
SONIC publishing remains to be integrated.

## Build on Linux

Install Git, a C++23 compiler (GCC 13+ or Clang 17+), CMake 3.24+ and Ninja.
From a new checkout:

```sh
git submodule update --init --recursive
cmake --preset release
cmake --build --preset release -j8
ctest --preset release
```

These commands require no weights, Python, sibling repositories, Nix, or network
access after fetching the pinned GGML submodule. The checked-in fixtures support
the model-free contract tests. Model-backed parity tests are separate.
Use `debug` instead of `release` for ASan/UBSan. Clang installations may require
a separate compiler-rt package for sanitizers.

For Vulkan, install the Vulkan loader/development headers, `glslc` and SPIR-V
headers (for example `libvulkan-dev glslc spirv-headers` on Debian/Ubuntu), then:

```sh
cmake --preset vulkan
cmake --build --preset vulkan -j8
ctest --preset vulkan
```

Select nonstandard SDK/toolchain locations with normal CMake variables such as
`CMAKE_PREFIX_PATH`, `CMAKE_CXX_COMPILER` and `Vulkan_GLSLC_EXECUTABLE`.
The [development guide](docs/DEVELOPMENT.md) covers installation, fuzzing,
independent builds and optional reference dependencies. Linux is the tested
platform; other operating systems are not claimed as validated.

The build sets `GGML_CPU_ALL_VARIANTS=ON` and `GGML_NATIVE=OFF`. It emits
portable CPU variants, including AVX2 and AVX-512 variants where supported by
the compiler, and GGML selects the best compatible module at runtime. Vulkan is
available through the `vulkan` preset. Keep compilation at eight jobs or fewer.

The Vulkan build includes a narrowly scoped, parity-checked ViTPose tile
selection for strict F32 live inference on RTX 5070 Ti. It operates on a
build-directory copy of GGML; the upstream submodule stays unchanged.
Set `GEMX_VITPOSE_TILES=0` at runtime or configure with
`-DGEMX_VITPOSE_TILES=OFF` to use upstream selection. See the
[tile experiment and validation](docs/VITPOSE-TILE-TUNING.md).
ViTPose now enables the validated flattening, normalization/SwiGLU fusions,
64×64 expansion tile and 64×64/128-thread QKV layout by default.
FlashAttention and packed gate/up remain disabled. See
[defaults and diagnostic opt-outs](docs/VITPOSE-DEFAULTS.md).

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

## Models and demo

Weights are separate from the source checkout. See the
[GGUF conversion model card](distribution/README.md) for the required files,
provenance, conversion commands, hashes and model-specific license terms.
This is a custom GEM-X runtime; these GGUFs are not LLMs for llama.cpp.

The optional demo needs Go 1.23+ and the native executable/backend/models:

```sh
(cd demo && GOMAXPROCS=8 CGO_ENABLED=0 go build -p 8 -o gemx-demo .)
./demo/gemx-demo --threads 8
```

Run from the repository root. The default paths select the `vulkan` build;
all model, backend, data and worker locations have command-line overrides.
Device selection defaults to backend device index 0, with an optional exact
name check. Offline Body processing additionally needs a compatible sam3d.cpp
installation, configured explicitly through its worker/model flags.
See the [demo guide](demo/README.md). The demo uses LocalAI branding and the
same sidebar/preview styling as sam3d.cpp. Live capture uses a single shared
camera/skeleton viewport and bounded preparation/upload lookahead.

## Validation and performance

The [current parity report](docs/LIVE-OFFLINE-PARITY.md) covers strict F32,
independent offline and live comparisons. Component fixtures extend through
120 frames; longer sequences use consecutive windows and do not reproduce
upstream's overlapping attention policy. No physical robot control is included.

On the recorded RTX 5070 Ti, strict-F32 ViTPose with flip augmentation takes
**48.92 ms** at batch two. The simulated-camera live demo with detection every
five processed frames reaches **15.35 fps**, including GEM and transport.
These are different measurement scopes; neither is a guarantee for other
hardware or cameras. See [optimization defaults](docs/VITPOSE-DEFAULTS.md) and
[live pipelining measurements](docs/LIVE-PIPELINING.md). Earlier fast-arithmetic
figures in dated experiment reports are historical, not the current strict
configuration.

The pinned upstream visual exemplar can be fetched with
`python scripts/download_e2e_exemplar.py`. NVIDIA publishes an annotated input
GIF plus in-camera and global-motion GIFs, but no clean source clip or numeric
motion. We use a lossless decode of that annotated input to run upstream and
native inference on identical pixels. Native offline output includes
`predictions.bin`, which `scripts/compare_e2e_parity.py` compares against
upstream raw heads. Use `scripts/audit_video_parity.py` for decoded skeleton
metrics; see [the current audit](docs/LIVE-OFFLINE-PARITY.md) for the measurements.

The [live profiling report](docs/LIVE-PROFILING.md) records strict-F32 stage
timings, byte-identical optimizations and remaining performance targets.

## License and provenance

Original contributions are Apache-2.0; third-party material retains its own
notices. Model weights have separate terms, including NVIDIA Open Model and
DINOv3 terms. See [LICENSE](LICENSE), [NOTICE](NOTICE) and
[licensing details](docs/LICENSING.md). No converted weights are checked in or
published by this audit.
