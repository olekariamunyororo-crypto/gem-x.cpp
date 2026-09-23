# Android Live Client for gem-x.cpp

Minimal CameraX + OkHttp client that streams phone camera frames into a running `gemx-demo` live session.

## Requirements

- Android Studio Ladybug / Koala or newer (AGP 8.7+)
- Android device or emulator with camera (API 26+)
- A machine running `./demo/gemx-demo --threads 8` reachable from the phone (same Wi-Fi or USB port-forward)

## Protocol used

```
POST /api/live?pipeline=2&detect_interval=5   → session id
PUT  /api/live/<id>/frame
     Header: X-GEMX-Frame: <1-based>
     Body:   JPEG
DELETE /api/live/<id>
```

## Quick start

1. Open this folder in Android Studio.
2. Edit `SERVER_BASE_URL` in `GemxConfig.kt` (e.g. `http://192.168.1.42:8098`).
3. Build & run on a physical device.
4. Grant camera permission → press **Start**.
5. On the desktop, open the gem-x browser UI or just watch the native worker process the frames.

## Notes

- Resolution is locked after the first successful frame.
- Uses `STRATEGY_KEEP_ONLY_LATEST` so the analyzer never queues up.
- JPEG quality defaults to 85; long edge is scaled to ≤ 960 px.
- The client keeps at most two frames in flight (matches server `pipeline=2`).

This is intentionally minimal — no pose visualisation on the phone yet.
