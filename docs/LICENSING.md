# Licensing and provenance

Original gem-x.cpp contributions are licensed under [Apache-2.0](../LICENSE),
except for third-party materials identified by [NOTICE](../NOTICE), file
headers and `LICENSES/`. This is not an assertion that every bundled component
or converted model is Apache-2.0.

| Material | Terms / source |
| --- | --- |
| Original native API, demo and conversion infrastructure | Apache-2.0 |
| GEM-X/SOMA research implementations and adaptations | NVIDIA Apache-2.0; pinned revisions in `reference/sources.json` |
| GGML and GGML-derived Vulkan modifications | MIT, `ggml/LICENSE` |
| DINOv3 material used by ViTPose | `LICENSES/DINOv3.md` |
| MHR geometry/identity data | `LICENSES/MHR-Apache-2.0.txt`, plus Momentum notices where applicable |
| OpenCV-derived crop arithmetic | `LICENSES/OpenCV-imgwarp.txt` |
| YOLOX inference / ByteTrack association adaptations | Apache-2.0 / MIT notices in `LICENSES/` |
| Bundled DevTools transport, adapted via sam3d.cpp | `LICENSES/Trellis2cpp-MIT.txt`, Copyright 2026 rms80 |
| LocalAI name and logo | Existing project branding, reused from sam3d.cpp; no third-party trademark grant or NVIDIA/Meta endorsement |

The root Apache license retains NVIDIA's notice; `NOTICE` identifies the
independent port and its contributors. Installations include the license and
attribution files. Source fixtures are reference outputs, not a grant to
redistribute separately licensed pretrained weights.

## Converted model weights

GGUF is a container format, not a license. NVIDIA's
[GEM-X model card](https://huggingface.co/nvidia/GEM-X) explicitly distinguishes
Apache-2.0 source from the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
for the model. NVIDIA's [SOMA-X model page](https://huggingface.co/nvidia/SOMA-X)
also identifies its model terms. DINOv3 and MHR components retain their additional
upstream notices. Optional SAM 3D Body weights are separate downloads governed
by Meta's SAM/DINOv3 terms; they are not included in the GEM-X conversion bundle.

The [conversion model card](../distribution/README.md) describes these distinct
files and dependencies. It is prepared for review, not a claim that weights
have been published. Before distribution, include the applicable license copies
and model notices, preserve attribution, and verify the identities of the exact
files being shipped. NVIDIA's model agreement requires the model notice
“Licensed by NVIDIA Corporation under the NVIDIA Open Model License”.
The prepared bundle notice includes it. Conversion does not remove upstream
usage or redistribution conditions.

Upstream license/source pages were checked on 2026-09-21. The pinned source and
asset revisions in `reference/sources.json` remain authoritative for conversion;
license-page links document terms rather than silently updating model inputs.

The official downloadable NVIDIA agreement (version October 24, 2025) is
preserved at `LICENSES/NVIDIA-Open-Model-2025-10-24.pdf`, SHA-256
`4d2fb590aa9b30c47f2058bff17291df7fa2aa0c1bd775a20703da9bb267cfab`.
It was retrieved from NVIDIA's linked license PDF, without modifying its text.
