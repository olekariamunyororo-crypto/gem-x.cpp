package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/png"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func testLiveApp(t *testing.T) *app {
	t.Helper()
	dir := t.TempDir()
	pipeline := filepath.Join(dir, "worker")
	// Exercise HTTP/worker lifecycle without GPU models; real rolling inference
	// is checked separately against the pinned video reference.
	script := `#!/bin/sh
printf 'READY\n'
n=0
while read command; do
 n=$((n+1))
 if test "$n" = 1; then printf 'WARMUP 1\n'; else
  printf GEMPOSE2 > "${11}/pose.gpose"
  printf 'POSE 1\n'
 fi
done
`
	if err := os.WriteFile(pipeline, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	a := &app{cfg: config{data: dir, pipeline: pipeline, threads: 8}}
	t.Cleanup(func() { a.stopLive() })
	return a
}
func liveRequest(a *app, method, path string, data []byte) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	a.route(w, httptest.NewRequest(method, path, bytes.NewReader(data)))
	return w
}
func startTestLive(t *testing.T, a *app) string {
	t.Helper()
	w := liveRequest(a, "POST", "/api/live", nil)
	if w.Code != 201 {
		t.Fatalf("start: %d %s", w.Code, w.Body)
	}
	var session struct{ ID string }
	if err := json.Unmarshal(w.Body.Bytes(), &session); err != nil {
		t.Fatal(err)
	}
	return "/api/live/" + session.ID
}
func livePNG(t *testing.T, width int) []byte {
	t.Helper()
	var b bytes.Buffer
	if err := png.Encode(&b, image.NewRGBA(image.Rect(0, 0, width, 16))); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}
func TestLiveLifecycle(t *testing.T) {
	a := testLiveApp(t)
	path := startTestLive(t, a)
	if w := liveRequest(a, "POST", "/api/live", nil); w.Code != 409 {
		t.Fatalf("second session: %d", w.Code)
	}
	if w := liveRequest(a, "PUT", path+"/frame", []byte("bad")); w.Code != 400 {
		t.Fatalf("bad image: %d", w.Code)
	}
	frame := livePNG(t, 16)
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 204 || w.Header().Get("X-GEMX-Sequence") != "1" {
		t.Fatalf("warmup: %d %s", w.Code, w.Body)
	}
	if w := liveRequest(a, "PUT", path+"/frame", livePNG(t, 24)); w.Code != 409 {
		t.Fatalf("dimension change: %d", w.Code)
	}
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 200 || w.Body.String() != "GEMPOSE2" || w.Header().Get("X-GEMX-Sequence") != "2" {
		t.Fatalf("pose: %d %s", w.Code, w.Body)
	}
	a.liveMu.Lock()
	s := a.live
	a.liveMu.Unlock()
	s.mu.Lock()
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 409 {
		t.Fatalf("overlap: %d", w.Code)
	}
	s.mu.Unlock()
	if w := liveRequest(a, "DELETE", path, nil); w.Code != 204 {
		t.Fatalf("stop: %d", w.Code)
	}
	if _, err := os.Stat(s.dir); !os.IsNotExist(err) {
		t.Fatalf("temporary live frames were retained: %v", err)
	}
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 404 {
		t.Fatalf("stale frame: %d", w.Code)
	}
	path = startTestLive(t, a)
	if w := liveRequest(a, "PUT", path+"/frame", frame); w.Code != 204 {
		t.Fatalf("restart did not warm up: %d", w.Code)
	}
	liveRequest(a, "DELETE", path, nil)
}
func TestLiveTimeoutReleasesWorker(t *testing.T) {
	a := testLiveApp(t)
	startTestLive(t, a)
	a.liveMu.Lock()
	s := a.live
	a.liveMu.Unlock()
	s.timer.Reset(time.Millisecond)
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		t.Fatal("idle worker survived timeout")
	}
	if !a.gpu.TryLock() {
		t.Fatal("GPU reservation survived timeout")
	}
	a.gpu.Unlock()
}

func TestLiveStartupDoesNotRequireBodyAssets(t *testing.T) {
	dir := t.TempDir()
	model := filepath.Join(dir, "present")
	if err := os.WriteFile(model, []byte("fixture"), 0600); err != nil {
		t.Fatal(err)
	}
	missing := filepath.Join(dir, "missing-body")
	c := config{data: dir, pipeline: model, denoiser: model, vitpose: model, yolox: model, module: model,
		bodyRunner: missing, bodyModule: missing, backbone: missing, branch: missing, mhr: missing,
		backend: "CPU", threads: 8, maxJobs: 1}
	if _, err := load(c); err != nil {
		t.Fatalf("live startup requires Body assets: %v", err)
	}
	c.vitpose = missing
	if _, err := load(c); err == nil {
		t.Fatal("missing live model accepted")
	}
}
