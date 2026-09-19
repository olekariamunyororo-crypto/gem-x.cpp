# GEM-X camera and video demo

Choose **Live webcam** to see continuous skeleton updates, or **Offline video**
to process a complete uploaded video or webcam recording and export a GLB.

Live mode follows upstream's camera example: one-frame person detection and
ViTPose, a 30-frame rolling GEM window, two-frame warm-up, and only the newest
pose displayed. It omits SAM3D Body, contact refinement and world trajectory.
The 3D view uses that frame's shape and gravity-aligned orientation. The camera
overlay uses the exact captured frame associated with the displayed pose.

Press **Start live**, grant camera permission, and keep the camera stationary
with your full body in view. You can choose a camera after opening it. The rate
cap limits capture; the displayed rate and frame age reflect actual processing.
Stop releases the worker and camera. Hiding the tab pauses live capture; restart
clears temporal context. Live frames/results are temporary and are removed on
stop or idle expiry. Offline and live inference share one GPU slot.

Camera access requires localhost or HTTPS. Remote plain HTTP can still process
uploaded videos. Live mode can run without SAM3D Body assets installed; those
assets are needed for offline jobs.

The offline path uses future context from the complete clip.

The native stages are:

1. YOLOX plus ByteTrack detects people, selects the dominant track, fills
   missing boxes and applies a symmetric five-frame smoothing window.
2. SAM 3D Body emits the 1024-value pose token expected by GEM-X for each
   tracked crop.
3. ViTPose produces the 77 two-dimensional keypoints.
4. GEM-X runs once over the full sequence, builds world- and camera-space SOMA
   skeletons, overlays the camera projection and ViTPose observations on the
   synchronized clip, and exports an animated GLB.

Clips are limited to 120 sampled frames, the context length covered by the
current parity work. The browser scales frames to at most 960 pixels on the
long edge before uploading them.

From the repository root:

```sh
cd demo
taskset -c 0-7 env GOMAXPROCS=8 CGO_ENABLED=0 go build -p 8 -o gemx-demo .
cd ..
taskset -c 0-7 ./demo/gemx-demo
```

All paths have local development defaults. Use the command-line flags when the
GEM-X and SAM3D repositories or model files are elsewhere. `--threads` is
validated to a maximum of eight.

The implementation follows and credits
[NVlabs/GEM-X](https://github.com/NVlabs/GEM-X) and uses the pose feature path
from [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body).

The default configuration uses strict F32 inference, F32 Body features and
upstream contact correction/grounding/IK. It requires
`generated/reference/gem-x-contact-f32.gguf`, converted with `--checkpoint`.
See [the parity report](../docs/LIVE-OFFLINE-PARITY.md) for conversion and
validation. `--strict=false --bf16 --contacts=false` selects the approximate
configuration; measurements from it should not be labelled strict parity.

The resident native worker is `gemx-pipeline --live-worker`; it uses the same
models and strict precision setting as the demo. The browser sends one frame
at a time, so slow inference cannot build an upload queue. The server rejects
concurrent frame requests, expires idle sessions after 30 seconds, and bounds
startup/inference to 120 seconds. A gap over two seconds resets native temporal
history. A stale pose disappears from the browser after two seconds.

Validation includes Go lifecycle tests, lossless 72-frame HTTP replay using
`scripts/qa_live_http.py`, and simulated-webcam Chromium checks using
`scripts/qa_live_browser.py`. The latter tests real native inference but does
not establish physical webcam quality or camera-to-robot latency. SONIC
publishing is not connected to this demo.

Recorded results: [live demo validation](../docs/live-demo-validation-2026-09-18.json).
