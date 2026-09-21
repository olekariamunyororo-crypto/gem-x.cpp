package main

import (
	"context"
	"crypto/rand"
	"embed"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

//go:embed web
var assets embed.FS

const maxFrames = 120
const maxFrameBytes = 5 << 20

var validID = regexp.MustCompile(`^[a-f0-9]{24}$`)

type config struct {
	addr, data, pipeline, denoiser, vitpose, yolox, module             string
	bodyRunner, bodyModule, backbone, branch, mhr, backend, deviceName string
	device, threads                                                    int
	maxJobs                                                            int
	bf16                                                               bool
	strict, contacts                                                   bool
}

type job struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	State    string    `json:"state"`
	Stage    string    `json:"stage"`
	Error    string    `json:"error,omitempty"`
	Created  time.Time `json:"created"`
	Finished time.Time `json:"finished,omitempty"`
	Width    int       `json:"width"`
	Height   int       `json:"height"`
	Frames   int       `json:"frames"`
	Received int       `json:"received"`
	FPS      float64   `json:"fps"`
}

type app struct {
	cfg    config
	gpu    sync.Mutex
	liveMu sync.Mutex
	live   *liveSession
	mu     sync.Mutex
	upload sync.Mutex
	jobs   map[string]*job
	queue  chan string
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func fail(w http.ResponseWriter, status int, err any) {
	writeJSON(w, status, map[string]any{"error": fmt.Sprint(err)})
}

func (a *app) dir(id string) string { return filepath.Join(a.cfg.data, id) }

func (a *app) save(j *job) error {
	b, err := json.MarshalIndent(j, "", "  ")
	if err != nil {
		return err
	}
	tmp := filepath.Join(a.dir(j.ID), "job.json.tmp")
	if err = os.WriteFile(tmp, append(b, '\n'), 0600); err != nil {
		return err
	}
	return os.Rename(tmp, filepath.Join(a.dir(j.ID), "job.json"))
}

func (a *app) stage(id, stage string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if j := a.jobs[id]; j != nil {
		j.Stage = stage
		_ = a.save(j)
	}
}

func (a *app) create(w http.ResponseWriter, r *http.Request) {
	var in struct {
		Name                  string `json:"name"`
		Width, Height, Frames int
		FPS                   float64 `json:"fps"`
	}
	dec := json.NewDecoder(io.LimitReader(r.Body, 4096))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&in); err != nil || dec.Decode(new(any)) != io.EOF {
		fail(w, 400, "invalid clip description")
		return
	}
	if in.Width < 8 || in.Height < 8 || in.Width > 4096 || in.Height > 4096 || int64(in.Width)*int64(in.Height) > 1_000_000 || in.Frames < 1 || in.Frames > maxFrames || in.FPS < 1 || in.FPS > 60 {
		fail(w, 400, "clip must be 1–120 frames, 1–60 fps, and at most 1 megapixel")
		return
	}
	if len(in.Name) > 160 {
		in.Name = in.Name[:160]
	}
	random := make([]byte, 12)
	if _, err := rand.Read(random); err != nil {
		fail(w, 500, err)
		return
	}
	j := &job{ID: hex.EncodeToString(random), Name: in.Name, State: "uploading", Stage: "Collecting the complete clip", Created: time.Now().UTC(), Width: in.Width, Height: in.Height, Frames: in.Frames, FPS: in.FPS}
	a.mu.Lock()
	if len(a.jobs) >= a.cfg.maxJobs {
		a.mu.Unlock()
		fail(w, 507, "job history is full; remove old job data before submitting another clip")
		return
	}
	if err := os.MkdirAll(filepath.Join(a.dir(j.ID), "frames"), 0700); err != nil {
		a.mu.Unlock()
		fail(w, 500, err)
		return
	}
	a.jobs[j.ID] = j
	err := a.save(j)
	a.mu.Unlock()
	if err != nil {
		fail(w, 500, err)
		return
	}
	writeJSON(w, 201, j)
}

func (a *app) frame(w http.ResponseWriter, r *http.Request, id, suffix string) {
	a.upload.Lock()
	defer a.upload.Unlock()
	index, err := strconv.Atoi(suffix)
	if err != nil || index < 0 || index >= maxFrames {
		fail(w, 404, "frame not found")
		return
	}
	a.mu.Lock()
	j := a.jobs[id]
	if j == nil || j.State != "uploading" || index != j.Received {
		a.mu.Unlock()
		fail(w, 409, "frames must be uploaded once, in order")
		return
	}
	a.mu.Unlock()
	r.Body = http.MaxBytesReader(w, r.Body, maxFrameBytes)
	data, err := io.ReadAll(r.Body)
	if err != nil || len(data) == 0 || len(data) > maxFrameBytes {
		fail(w, 413, "frame exceeds 5 MiB")
		return
	}
	im, _, err := decodeImage(data)
	if err != nil {
		fail(w, 400, err)
		return
	}
	if im.Bounds().Dx() != j.Width || im.Bounds().Dy() != j.Height {
		fail(w, 400, "all clip frames must have the declared dimensions")
		return
	}
	path := filepath.Join(a.dir(id), "frames", fmt.Sprintf("%06d.input", index))
	if err = os.WriteFile(path, packImage(im), 0600); err != nil {
		fail(w, 500, err)
		return
	}
	if index == 0 {
		_ = os.WriteFile(filepath.Join(a.dir(id), "preview.jpg"), data, 0600)
	}
	a.mu.Lock()
	j.Received++
	_ = a.save(j)
	copy := *j
	a.mu.Unlock()
	writeJSON(w, 200, &copy)
}

func (a *app) start(w http.ResponseWriter, r *http.Request, id string) {
	a.mu.Lock()
	j := a.jobs[id]
	if j == nil || j.State != "uploading" || j.Received != j.Frames {
		a.mu.Unlock()
		fail(w, 409, "the complete clip must be uploaded before processing")
		return
	}
	j.State, j.Stage = "queued", "Waiting for the offline inference worker"
	_ = a.save(j)
	copy := *j
	a.mu.Unlock()
	select {
	case a.queue <- id:
		writeJSON(w, 202, &copy)
	default:
		a.mu.Lock()
		j.State, j.Stage = "uploading", "Queue full; try again"
		_ = a.save(j)
		a.mu.Unlock()
		fail(w, 429, "inference queue is full")
	}
}

func (a *app) status(w http.ResponseWriter, id string) {
	a.mu.Lock()
	j := a.jobs[id]
	if j == nil {
		a.mu.Unlock()
		fail(w, 404, "job not found")
		return
	}
	copy := *j
	a.mu.Unlock()
	writeJSON(w, 200, &copy)
}

func (a *app) route(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	parts := strings.Split(strings.Trim(r.URL.Path, "/"), "/")
	if len(parts) >= 2 && parts[0] == "api" && parts[1] == "live" {
		a.liveRoute(w, r, parts)
		return
	}
	if r.URL.Path == "/api/jobs" && r.Method == http.MethodPost {
		a.create(w, r)
		return
	}
	if len(parts) < 3 || parts[0] != "api" || parts[1] != "jobs" || !validID.MatchString(parts[2]) {
		fail(w, 404, "not found")
		return
	}
	id := parts[2]
	if len(parts) == 3 && r.Method == http.MethodGet {
		a.status(w, id)
		return
	}
	if len(parts) == 4 && parts[3] == "process" && r.Method == http.MethodPost {
		a.start(w, r, id)
		return
	}
	if len(parts) == 5 && parts[3] == "frames" && r.Method == http.MethodPut {
		a.frame(w, r, id, parts[4])
		return
	}
	if len(parts) == 5 && parts[3] == "poses" && r.Method == http.MethodGet {
		if _, err := strconv.Atoi(parts[4]); err != nil {
			fail(w, 404, "pose not found")
			return
		}
		http.ServeFile(w, r, filepath.Join(a.dir(id), "output", parts[4]+".gpose"))
		return
	}
	if len(parts) == 4 && parts[3] == "motion.glb" && r.Method == http.MethodGet {
		w.Header().Set("Content-Disposition", `attachment; filename="gem-x-motion.glb"`)
		http.ServeFile(w, r, filepath.Join(a.dir(id), "output", "motion.glb"))
		return
	}
	fail(w, 404, "not found")
}

func load(c config) (*app, error) {
	if c.threads < 1 || c.threads > 8 || c.device < 0 || c.maxJobs < 1 || (c.backend != "CPU" && c.backend != "Vulkan") {
		return nil, fmt.Errorf("threads must be 1–8 and backend CPU or Vulkan")
	}
	for name, path := range map[string]string{"pipeline": c.pipeline, "denoiser": c.denoiser, "vitpose": c.vitpose, "yolox": c.yolox, "module": c.module, "body-runner": c.bodyRunner, "body-module": c.bodyModule, "backbone": c.backbone, "branch": c.branch, "mhr": c.mhr} {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return nil, err
		}
		// Body assets are only needed when an offline job actually runs.
		bodyAsset := name == "body-runner" || name == "body-module" || name == "backbone" || name == "branch" || name == "mhr"
		if info, err := os.Stat(absolute); !bodyAsset && (err != nil || (info.IsDir() && name != "module")) {
			return nil, fmt.Errorf("%s file is unavailable: %s", name, absolute)
		}
		switch name {
		case "pipeline":
			c.pipeline = absolute
		case "denoiser":
			c.denoiser = absolute
		case "vitpose":
			c.vitpose = absolute
		case "yolox":
			c.yolox = absolute
		case "module":
			c.module = absolute
		case "body-runner":
			c.bodyRunner = absolute
		case "body-module":
			c.bodyModule = absolute
		case "backbone":
			c.backbone = absolute
		case "branch":
			c.branch = absolute
		case "mhr":
			c.mhr = absolute
		}
	}
	c.data, _ = filepath.Abs(c.data)
	if err := os.MkdirAll(c.data, 0700); err != nil {
		return nil, err
	}
	a := &app{cfg: c, jobs: map[string]*job{}, queue: make(chan string, 2)}
	entries, _ := os.ReadDir(c.data)
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		b, err := os.ReadFile(filepath.Join(c.data, entry.Name(), "job.json"))
		if err != nil {
			continue
		}
		var j job
		if json.Unmarshal(b, &j) == nil && j.ID == entry.Name() {
			if j.State == "running" || j.State == "queued" {
				j.State = "failed"
				j.Error = "server stopped during inference"
				_ = a.save(&j)
			}
			a.jobs[j.ID] = &j
		}
	}
	return a, nil
}

func main() {
	runtime.GOMAXPROCS(min(runtime.GOMAXPROCS(0), 8))
	var c config
	flag.StringVar(&c.addr, "listen", "127.0.0.1:8098", "HTTP listen address")
	flag.StringVar(&c.data, "data", "generated/demo", "job directory")
	flag.StringVar(&c.pipeline, "pipeline", "build/vulkan/gemx-pipeline", "GEM-X pipeline executable")
	flag.StringVar(&c.denoiser, "denoiser", "generated/reference/gem-x-contact-f32.gguf", "GEM-X temporal model (convert with --checkpoint for contact support)")
	flag.StringVar(&c.vitpose, "vitpose", "generated/reference/vitpose-f32.gguf", "ViTPose model")
	flag.StringVar(&c.yolox, "yolox", "generated/reference/yolox-f32.gguf", "YOLOX model")
	flag.StringVar(&c.module, "module", "build/vulkan/bin/libggml-vulkan.so", "GEM-X GGML backend module")
	flag.StringVar(&c.bodyRunner, "body-runner", "../sam3d.cpp/build/vulkan-bf16-performance/bin/sam3d-body-infer", "SAM3D Body worker")
	flag.StringVar(&c.bodyModule, "body-module", "../sam3d.cpp/build/vulkan-bf16-performance/bin/libggml-vulkan.so", "SAM3D Body GGML backend module")
	flag.StringVar(&c.backbone, "backbone", "../sam3d.cpp/generated/models/sam-3d-body-dinov3/body-dinov3-f32.gguf", "SAM3D Body backbone")
	flag.StringVar(&c.branch, "branch", "../sam3d.cpp/generated/models/sam-3d-body-dinov3/body-pose-branch-f32.gguf", "SAM3D Body pose branch")
	flag.StringVar(&c.mhr, "mhr", "../sam3d.cpp/generated/models/mhr-public/mhr-lod1-f32.gguf", "SAM3D Body MHR model")
	flag.StringVar(&c.backend, "backend", "Vulkan", "CPU or Vulkan")
	flag.StringVar(&c.deviceName, "device-name", "-", "exact GGML device name, or -")
	flag.IntVar(&c.device, "device", 0, "backend device")
	flag.IntVar(&c.threads, "threads", 8, "CPU threads, maximum 8")
	flag.IntVar(&c.maxJobs, "max-jobs", 20, "maximum retained jobs")
	flag.BoolVar(&c.bf16, "bf16", false, "use approximate BF16 SAM3D Body backbone")
	flag.BoolVar(&c.strict, "strict", true, "use strict F32 Vulkan inference for detector, ViTPose and GEM")
	flag.BoolVar(&c.contacts, "contacts", true, "apply upstream offline contact correction, grounding and IK")
	flag.Parse()
	a, err := load(c)
	if err != nil {
		log.Fatal(err)
	}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	go a.work(ctx)
	web, _ := fs.Sub(assets, "web")
	mux := http.NewServeMux()
	mux.HandleFunc("/api/", a.route)
	mux.Handle("/", http.FileServer(http.FS(web)))
	server := &http.Server{Addr: c.addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second, WriteTimeout: 30 * time.Minute, IdleTimeout: 60 * time.Second}
	go func() {
		<-ctx.Done()
		a.stopLive()
		shutdownContext, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = server.Shutdown(shutdownContext)
	}()
	log.Printf("GEM-X live/offline demo listening on http://%s", c.addr)
	if err = server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
