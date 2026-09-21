# Portability and release audit — 2026-09-21

## Independent build

A new checkout was created with `git clone --no-local` from committed source;
the pinned GGML submodule was fetched independently from its public GitHub URL.
No `generated/`, old build directories, weights, Nix store, sibling repositories
or host toolchain were mounted into the build container. Network access was
disabled during builds/tests. All build/runtime processes used CPU affinity 0–7
and at most eight build jobs. The initial audited revision was `0a55308`; the
checkout was then advanced to `c9a12cb` and all build/test checks repeated.

The stock Debian 13 toolchain was:

| Tool | Version |
| --- | --- |
| GCC | 14.2.0 |
| Clang/compiler-rt | 19.1.7 |
| CMake | 3.31.6 |
| Ninja | 1.12.1 |
| Go | 1.24.4 |
| Vulkan headers/loader development package | 1.4.309.0 |
| glslc | Debian package 2025.2-1 |
| GGML | `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226` |

CPU `release` and `vulkan` both configure and build successfully; all six
model-free CTest contracts pass in each. A separate Clang ASan/UBSan build also
passes all six. The fresh-checkout demo passes Go race tests and builds with
CGO disabled. The installed native CLI resolves its libraries from the install
prefix, with no missing dependencies, and runs its argument validation.
The build includes portable CPU variants rather than compiling only for the
build machine's CPU. The Vulkan build check does not imply GPU inference was
run inside the generic container.

`tools/audit.Dockerfile` reproduces the stock toolchain. Typical independent
validation, with a fresh checkout mounted at `/work`:

```sh
docker build -t gemx-build-audit -f tools/audit.Dockerfile .
docker run --rm --cpuset-cpus 0-7 --network none \
  -v /path/to/fresh-checkout:/work gemx-build-audit sh -c '
    cmake --preset release && cmake --build --preset release -j8 && ctest --preset release &&
    cmake --preset vulkan && cmake --build --preset vulkan -j8 && ctest --preset vulkan &&
    cd demo && go test -race -p 8 ./... && CGO_ENABLED=0 go build -p 8 .'
```

Adjust CPU IDs to those available on the host, keeping at most eight. Docker
is an optional audit tool, not a build requirement. Native instructions in the
README use ordinary CMake commands. The CI workflow runs equivalent stock
Debian native/fuzz jobs plus a Go job; no remote CI execution is claimed here.

## Non-GGUF input fuzzing

Seeded campaigns ran with no crashes, sanitizer findings or failing inputs:

| Surface | Engine | Budget | Executions |
| --- | --- | ---: | ---: |
| Packed RGB, Body tensor records, poses, all manifest versions, CLI numbers | Clang libFuzzer + ASan/UBSan | 60 s | 4,069,512 |
| Observation/camera arrays, rotations/motion, RGB strides/capacities and ViTPose/YOLOX preparation | Clang libFuzzer + ASan/UBSan | 60 s | 2,193 |
| JPEG/PNG decode and RGB packing | Go fuzz, 4 workers | 30 s | 3,060,601 |
| Packed-image box patching | Go fuzz, 4 workers | 30 s | 3,024,610 |
| Job JSON | Go fuzz, 4 workers | 30 s | 386,623 |
| Detector box records/counts | Go fuzz, 4 workers | 30 s | 3,025,889 |

The API target is slower because valid cases execute image preprocessing.
These counts are bounded campaign evidence, not complete coverage or proof of
safety. Native GGUF parsing, external conversion libraries, browser video codec
implementations, arbitrary invalid C pointers and model-dependent export math
are outside these campaigns. Production parsers, not duplicate implementations,
are exercised; fuzzer-supplied paths are never opened. Reproduction commands
and exact scope are in [DEVELOPMENT.md](DEVELOPMENT.md).

The code review additionally fixed empty numeric argument acceptance, trailing
export manifest acceptance, nonfinite sequence boxes, and unbounded/invalid
counts or nonfinite values in detector responses. Regression tests cover the
numeric/trailing/truncated input contracts and backend directory selection.
No change was made to model math or precision.

## Portability, docs and packaging

- GPU name matching is optional in the demo and audit/replay tools; index 0 is
  the generic default. GPU-specific kernel guards remain deliberate restrictions.
- Model/backend/data/worker locations are configurable. The offline audit's
  SAM3D build directory and profiler container/device/metric set are configurable.
  CUDA comparison defaults now point to repository assets instead of fixed
  temporary files. Browser QA includes its own attributed DevTools helper.
- The demo accepts a CPU backend directory and honors a lower `GOMAXPROCS` while
  retaining the eight-thread ceiling. The unsupported no-GGML configuration now
  fails at configure time with an explicit explanation.
- README build instructions start with public submodule initialization and
  standard tools. Obsolete fast-arithmetic numbers were removed from the current
  summary. Historical experiment records are labelled and retain their original
  machine paths as provenance rather than runnable defaults.
- Installations include original/third-party notices, including GGML MIT.
  The model card, tested GGUF hashes, offline verifier, model notice and official
  NVIDIA model-license PDF are prepared under `distribution/` and `LICENSES/`.
  All three GGUF identities passed the offline verifier. No weights were uploaded.

## Demo and parity

The LocalAI UI follows sam3d.cpp's dark-blue sidebar, controls and preview pane,
using the same logo asset. It retains the shared raw-camera/skeleton viewport.
Real-inference Chromium checks cover desktop and mobile layout, loaded branding,
stop-to-camera, close/restart, offline-mode switching, permission denial and the
two-request bound, without JavaScript exceptions. The mobile viewport had no
horizontal overflow. The same checks passed on the deployed HTTPS devroute. These use a simulated
webcam, not a physical camera.

After the native parser refactor and generic device selection, the 144-frame
strict-F32 live replay emitted **143 byte-identical poses** against the accepted
profile's complete pose hashes. This preserves the existing upstream parity scope; it does not establish
new upstream or physical robot validation. See LIVE-OFFLINE-PARITY.md for limits.

Raw build/fuzz logs, toolchain versions, parity report and screenshots are kept
locally under `generated/release-audit/`; generated artifacts and weights remain
excluded from the repository. A compact machine-readable record accompanies this
audit at `release-audit-2026-09-21.json`.
