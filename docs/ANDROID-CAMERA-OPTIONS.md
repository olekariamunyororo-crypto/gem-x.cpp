# Using an Android phone as a camera source for gem-x.cpp

gem-x.cpp currently expects either:
- a browser webcam (Live mode), or
- uploaded / recorded video (Offline mode).

An Android phone can be used in two practical ways today, and a third longer-term path exists.

## 1. Easiest path today — Android as virtual webcam on Linux

Turn the phone into a V4L2 virtual camera that the existing browser demo (or any V4L2 client) can open.

**Recommended options (2026):**

| Tool | Connection | Notes |
|------|------------|-------|
| **scrcpy** + v4l2loopback | USB (preferred) or Wi-Fi | Official, low latency, `--video-source=camera` |
| **DroidCam** | USB / Wi-Fi | Mature, dedicated client on Linux |
| **IP Webcam** (Play Store) + ffmpeg | Wi-Fi | Simple HTTP MJPEG stream → v4l2loopback |
| **AVream / adbcam** | USB / Wi-Fi | Newer wrappers around scrcpy + v4l2loopback |

### Quick scrcpy example
```bash
# One-time setup
sudo apt install scrcpy v4l2loopback-dkms android-tools-adb
sudo modprobe v4l2loopback exclusive_caps=1 card_label="Android Camera"

# Stream phone camera into /dev/videoX
scrcpy --video-source=camera --camera-facing=front --no-audio \
       --v4l2-sink=/dev/video0
```
Then open the gem-x demo in the browser and select the new virtual camera (or let the browser pick it).

This requires **zero changes** to gem-x.cpp and works with the existing live pipeline.

## 2. Medium effort — HTTP / WebSocket frame push from Android

Write a small Android app (CameraX `ImageAnalysis`) that:
1. Captures frames at the resolution the demo expects (or scales them).
2. JPEG-encodes or sends raw RGB/YUV over HTTP/WebSocket to the existing `gemx-demo` server.

The current demo already accepts uploaded frames; a small protocol extension would allow continuous live push. This keeps inference on the powerful desktop/GPU machine while using the phone only as a high-quality camera.

## 3. Long-term / research — native Android build

Running the full GGML + Vulkan pipeline on-device is possible in principle but non-trivial:

- **GGML** has Android / NDK support and Vulkan backend.
- Camera frames can be obtained zero-copy via `AHardwareBuffer` → Vulkan external memory (see ncnn Android Hardware Buffer docs, CameraX + NDK samples).
- Challenges: model size (ViTPose + GEM-X + YOLOX), thermal limits, battery, and the current demo being tightly coupled to a desktop Go server + browser UI.

Realistic first milestone would be a minimal native activity that runs YOLOX + ViTPose only and streams keypoints, leaving the temporal GEM denoiser on a more powerful device.

## Recommendation for most users

Start with **option 1** (scrcpy / DroidCam / IP Webcam → v4l2loopback). It gives full-body pose tracking with a phone camera in minutes and re-uses every performance improvement already in the repo.

Further native Android work can be explored once the desktop path is solid and measured.
