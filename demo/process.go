package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func writePathManifest(path, magic string, rows [][]string) error {
	var out bytes.Buffer
	out.WriteString(magic)
	if err := binary.Write(&out, binary.LittleEndian, uint32(len(rows))); err != nil {
		return err
	}
	for _, row := range rows {
		for _, value := range row {
			if len(value) < 1 || len(value) > 4096 || strings.ContainsRune(value, 0) {
				return fmt.Errorf("invalid manifest path")
			}
			_ = binary.Write(&out, binary.LittleEndian, uint32(len(value)))
			out.WriteString(value)
		}
	}
	return os.WriteFile(path, out.Bytes(), 0600)
}

func readBoxes(path string, count int) ([][4]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b) != 12+count*16 || string(b[:8]) != "GEMBOX01" || int(binary.LittleEndian.Uint32(b[8:12])) != count {
		return nil, fmt.Errorf("invalid detector box output")
	}
	boxes := make([][4]float32, count)
	at := 12
	for i := range boxes {
		for axis := range boxes[i] {
			boxes[i][axis] = math.Float32frombits(binary.LittleEndian.Uint32(b[at:]))
			at += 4
		}
	}
	return boxes, nil
}

func writeSequenceManifest(path string, rows [][]string, boxes [][4]float32) error {
	if len(rows) != len(boxes) {
		return fmt.Errorf("sequence box count mismatch")
	}
	if err := writePathManifest(path, "GEMSEQ02", rows); err != nil {
		return err
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, box := range boxes {
		if err := binary.Write(f, binary.LittleEndian, observationBox(box)); err != nil {
			return err
		}
	}
	return nil
}

func (a *app) native(ctx context.Context, id string, args ...string) error {
	cmd := exec.CommandContext(ctx, a.cfg.pipeline, args...)
	cmd.Env = os.Environ()
	if a.cfg.strict {
		precision := a.cfg
		precision.bf16 = false
		cmd.Env = precision.bodyEnvironment(cmd.Env)
	}
	out, err := cmd.CombinedOutput()
	logPath := filepath.Join(a.dir(id), "inference.log")
	f, openErr := os.OpenFile(logPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if openErr == nil {
		_, _ = fmt.Fprintf(f, "$ %s %s\n%s\n", filepath.Base(a.cfg.pipeline), strings.Join(args, " "), out)
		_ = f.Close()
	}
	if err != nil {
		return fmt.Errorf("%s: %w: %s", args[0], err, strings.TrimSpace(string(out)))
	}
	return nil
}

func (a *app) process(ctx context.Context, id string) error {
	a.mu.Lock()
	j := a.jobs[id]
	snapshot := *j
	a.mu.Unlock()
	dir := a.dir(id)
	frameRows := make([][]string, snapshot.Frames)
	for i := range frameRows {
		frameRows[i] = []string{filepath.Join(dir, "frames", fmt.Sprintf("%06d.input", i))}
	}
	images := filepath.Join(dir, "images.manifest")
	if err := writePathManifest(images, "GEMIMGS1", frameRows); err != nil {
		return err
	}
	a.stage(id, "Detecting and tracking the primary person across the clip")
	c := a.cfg
	if err := a.native(ctx, id, "--detect", c.yolox, c.module, c.backend, strconv.Itoa(c.device), c.deviceName, strconv.Itoa(c.threads), images, filepath.Join(dir, "boxes.bin")); err != nil {
		return err
	}
	boxes, err := readBoxes(filepath.Join(dir, "boxes.bin"), snapshot.Frames)
	if err != nil {
		return err
	}
	for i, box := range boxes {
		path := frameRows[i][0]
		packed, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		if err = patchPackedBox(packed, bodyBox(box)); err != nil {
			return err
		}
		if err = os.WriteFile(path, packed, 0600); err != nil {
			return err
		}
	}
	a.stage(id, "Extracting SAM3D Body pose features for every frame")
	bodyDir := filepath.Join(dir, "body")
	if err = os.MkdirAll(bodyDir, 0700); err != nil {
		return err
	}
	bodyPaths, err := a.runBodySequence(ctx, id, frameRows, bodyDir)
	if err != nil {
		return err
	}
	sequenceRows := make([][]string, snapshot.Frames)
	for i := range sequenceRows {
		sequenceRows[i] = []string{frameRows[i][0], bodyPaths[i]}
	}
	sequence := filepath.Join(dir, "sequence.manifest")
	if err = writeSequenceManifest(sequence, sequenceRows, boxes); err != nil {
		return err
	}
	output := filepath.Join(dir, "output")
	if err = os.MkdirAll(output, 0700); err != nil {
		return err
	}
	a.stage(id, "Running full-sequence GEM-X motion inference")
	mode := "--offline"
	if c.contacts {
		mode = "--offline-contact"
	}
	return a.native(ctx, id, mode, c.denoiser, c.vitpose, c.module, c.backend, strconv.Itoa(c.device), c.deviceName, strconv.Itoa(c.threads), sequence, output, strconv.FormatFloat(snapshot.FPS, 'g', -1, 64))
}

func (a *app) work(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case id := <-a.queue:
			a.mu.Lock()
			j := a.jobs[id]
			if j == nil {
				a.mu.Unlock()
				continue
			}
			j.State = "running"
			j.Stage = "Preparing the complete clip"
			_ = a.save(j)
			a.mu.Unlock()
			jobCtx, cancel := context.WithTimeout(ctx, 30*time.Minute)
			a.gpu.Lock()
			err := a.process(jobCtx, id)
			a.gpu.Unlock()
			cancel()
			a.mu.Lock()
			j = a.jobs[id]
			j.Finished = time.Now().UTC()
			if err != nil {
				j.State = "failed"
				j.Stage = "Inference failed"
				j.Error = err.Error()
			} else {
				j.State = "complete"
				j.Stage = "Animated skeleton ready"
			}
			_ = a.save(j)
			a.mu.Unlock()
		}
	}
}
