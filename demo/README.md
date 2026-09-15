# GEM-X offline demo

This demo accepts an uploaded video or a clip recorded in the browser. It
collects the complete clip before inference and deliberately has no live
GEM-X mode. Whole-sequence processing preserves the future context used by the
published GEM-X pipeline.

The native stages are:

1. YOLOX plus ByteTrack detects people, selects the dominant track, fills
   missing boxes and applies a symmetric five-frame smoothing window.
2. SAM 3D Body emits the 1024-value pose token expected by GEM-X for each
   tracked crop.
3. ViTPose produces the 77 two-dimensional keypoints.
4. GEM-X runs once over the full sequence, builds the SOMA skeleton, and
   exports per-frame pose files plus an animated GLB.

Clips are limited to 120 sampled frames, the context length covered by the
current parity work. The browser scales frames to at most 960 pixels on the
long edge before uploading them.

From the repository root:

```sh
cd demo
CGO_ENABLED=0 go build -o gemx-demo .
cd ..
taskset -c 0-7 ./demo/gemx-demo
```

All paths have local development defaults. Use the command-line flags when the
GEM-X and SAM3D repositories or model files are elsewhere. `--threads` is
validated to a maximum of eight.

The implementation follows and credits
[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) and uses the pose feature path
from [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body).
