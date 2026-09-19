package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

// A live session owns the GPU until stopped or idle. HTTP frame requests are
// serialized without queuing; model state stays in one native process.
type liveSession struct {
	id, dir                 string
	width, height, sequence int
	mu                      sync.Mutex
	cancel                  context.CancelFunc
	timer                   *time.Timer
	input                   io.WriteCloser
	output                  *bufio.Reader
	done                    chan struct{}
	log                     liveLog
}

type liveLog struct {
	mu   sync.Mutex
	text []byte
}

func (l *liveLog) Write(p []byte) (int, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := len(p)
	l.text = append(l.text, p...)
	if len(l.text) > 65536 {
		l.text = append([]byte(nil), l.text[len(l.text)-65536:]...)
	}
	return n, nil
}
func (l *liveLog) String() string { l.mu.Lock(); defer l.mu.Unlock(); return string(l.text) }

func (a *app) liveStart(w http.ResponseWriter, r *http.Request) {
	if !a.gpu.TryLock() {
		fail(w, 409, "inference is busy; stop the current live session or wait for the clip to finish")
		return
	}
	owned := true
	defer func() {
		if owned {
			a.gpu.Unlock()
		}
	}()
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		fail(w, 500, err)
		return
	}
	dir, err := os.MkdirTemp(a.cfg.data, "live-")
	if err != nil {
		fail(w, 500, err)
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	s := &liveSession{id: hex.EncodeToString(random), dir: dir, cancel: cancel, done: make(chan struct{})}
	c := a.cfg
	cmd := exec.CommandContext(ctx, c.pipeline, "--live-worker", c.denoiser, c.vitpose, c.yolox, c.module, c.backend, strconv.Itoa(c.device), c.deviceName, strconv.Itoa(c.threads), "30", dir)
	cmd.Env = os.Environ()
	if c.strict {
		precision := c
		precision.bf16 = false
		cmd.Env = precision.bodyEnvironment(cmd.Env)
	}
	cmd.Stderr = &s.log
	input, err := cmd.StdinPipe()
	if err == nil {
		var output io.ReadCloser
		output, err = cmd.StdoutPipe()
		s.output = bufio.NewReaderSize(output, 4096)
	}
	if err == nil {
		err = cmd.Start()
	}
	if err != nil {
		cancel()
		os.RemoveAll(dir)
		fail(w, 500, err)
		return
	}
	s.input = input
	s.timer = time.AfterFunc(120*time.Second, cancel)
	a.liveMu.Lock()
	a.live = s
	a.liveMu.Unlock()
	owned = false
	go func() {
		_ = cmd.Wait()
		cancel()
		s.timer.Stop()
		// A frame handler may still be reading its output file.
		s.mu.Lock()
		_ = os.RemoveAll(dir)
		s.mu.Unlock()
		a.liveMu.Lock()
		if a.live == s {
			a.live = nil
		}
		a.liveMu.Unlock()
		a.gpu.Unlock()
		close(s.done)
	}()
	stopDisconnect := context.AfterFunc(r.Context(), cancel)
	defer stopDisconnect()
	line, err := s.output.ReadString('\n')
	if err != nil || strings.TrimSpace(line) != "READY" {
		cancel()
		fail(w, 500, "live worker failed to start: "+s.log.String())
		return
	}
	s.timer.Reset(30 * time.Second)
	writeJSON(w, 201, map[string]any{"id": s.id, "window": 30})
}

func (a *app) liveRoute(w http.ResponseWriter, r *http.Request, parts []string) {
	w.Header().Set("Cache-Control", "no-store")
	if len(parts) == 2 && r.Method == http.MethodPost {
		a.liveStart(w, r)
		return
	}
	if len(parts) < 3 || !validID.MatchString(parts[2]) {
		fail(w, 404, "live session not found")
		return
	}
	a.liveMu.Lock()
	s := a.live
	a.liveMu.Unlock()
	if s == nil || s.id != parts[2] {
		fail(w, 404, "live session ended; start again")
		return
	}
	if len(parts) == 3 && r.Method == http.MethodDelete {
		s.cancel()
		select {
		case <-s.done:
		case <-r.Context().Done():
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if len(parts) != 4 || parts[3] != "frame" || r.Method != http.MethodPut {
		fail(w, 404, "not found")
		return
	}
	if !s.mu.TryLock() {
		fail(w, 409, "a camera frame is already being processed")
		return
	}
	defer s.mu.Unlock()
	select {
	case <-s.done:
		fail(w, 410, "live session ended")
		return
	default:
	}
	s.timer.Reset(120 * time.Second)
	defer s.timer.Reset(30 * time.Second)
	r.Body = http.MaxBytesReader(w, r.Body, maxFrameBytes)
	data, err := io.ReadAll(r.Body)
	if err != nil {
		fail(w, 413, "frame exceeds 5 MiB")
		return
	}
	im, _, err := decodeImage(data)
	if err != nil {
		fail(w, 400, err)
		return
	}
	width, height := im.Bounds().Dx(), im.Bounds().Dy()
	if s.width != 0 && (s.width != width || s.height != height) {
		fail(w, 409, "camera dimensions changed; restart live capture")
		return
	}
	s.width, s.height = width, height
	if err = os.WriteFile(filepath.Join(s.dir, "frame.input"), packImage(im), 0600); err != nil {
		fail(w, 500, err)
		return
	}
	stopDisconnect := context.AfterFunc(r.Context(), s.cancel)
	defer stopDisconnect()
	started := time.Now()
	if _, err = fmt.Fprintln(s.input, "FRAME"); err != nil {
		s.cancel()
		fail(w, 500, "live worker stopped: "+s.log.String())
		return
	}
	line, err := s.output.ReadString('\n')
	if err != nil {
		s.cancel()
		fail(w, 500, "live inference failed: "+s.log.String())
		return
	}
	fields := strings.Fields(line)
	if len(fields) != 2 || (fields[0] != "WARMUP" && fields[0] != "POSE") {
		s.cancel()
		fail(w, 500, "invalid live worker reply")
		return
	}
	s.sequence++
	w.Header().Set("X-GEMX-Sequence", strconv.Itoa(s.sequence))
	w.Header().Set("X-GEMX-People", fields[1])
	w.Header().Set("X-GEMX-Inference-Ms", strconv.FormatInt(time.Since(started).Milliseconds(), 10))
	if fields[0] == "WARMUP" {
		w.WriteHeader(http.StatusNoContent)
		return
	}
	pose, err := os.ReadFile(filepath.Join(s.dir, "pose.gpose"))
	if err != nil {
		s.cancel()
		fail(w, 500, err)
		return
	}
	w.Header().Set("Content-Type", "application/octet-stream")
	_, _ = w.Write(pose)
}

func (a *app) stopLive() {
	a.liveMu.Lock()
	defer a.liveMu.Unlock()
	if a.live != nil {
		a.live.cancel()
	}
}
