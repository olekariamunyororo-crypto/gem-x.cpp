package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

func (c config) bodyEnvironment(inherited []string) []string {
	owned := map[string]bool{
		"GGML_VK_F32_NARROW_MATMUL": true, "GGML_VK_F32_NARROW_TRACE": true, "GGML_VK_F32_NARROW_TILE": true,
		"SAM3D_BF16_PACKED_FFN": true, "GGML_VK_FUSE_BF16_SILU_GATE": true, "GGML_VK_FUSE_BF16_AFFINE": true,
		"GGML_VK_FUSE_BF16_NORM_AFFINE": true, "GGML_VK_F32_MATVEC_ROWS": true, "GGML_VK_F32_MATVEC_TRACE": true,
		"SAM3D_SIMD_SKINNING": true, "SAM3D_SIMD_IMAGE": true, "SAM3D_IMAGE_GATHER": true, "SAM3D_BATCHED_TRANSFERS": true,
		"GGML_VK_BF16_MATMUL_TILE": true, "GGML_VK_BF16_MATMUL_TRACE": true, "GGML_VK_DISABLE_F16": true,
		"GGML_VK_DISABLE_COOPMAT": true, "GGML_VK_DISABLE_COOPMAT2": true, "GGML_VK_FUSE_BF16_ROUND": true,
		"GGML_VK_FUSE_BF16_BINARY": true, "GGML_VK_BF16_BINARY_LINEAR": true, "SAM3D_BF16_COOPMAT2": true,
		"SAM3D_BF16_FLASH_ATTENTION": true, "SAM3D_BF16_PRECISE_PREFIX": true, "SAM3D_BF16_PREFIX_PAD": true,
		"SAM3D_BF16_PREFIX_MV": true, "SAM3D_BF16_PREFIX_TRANSPOSE": true,
	}
	out := make([]string, 0, len(inherited)+14)
	for _, entry := range inherited {
		key, _, _ := strings.Cut(entry, "=")
		if !owned[key] {
			out = append(out, entry)
		}
	}
	out = append(out, "GGML_VK_DISABLE_F16=1")
	if c.bf16 && c.backend == "Vulkan" {
		return append(out, "SAM3D_BF16_COOPMAT2=1", "SAM3D_BF16_FLASH_ATTENTION=1", "GGML_VK_FUSE_BF16_SILU_GATE=1", "GGML_VK_FUSE_BF16_AFFINE=1", "GGML_VK_FUSE_BF16_NORM_AFFINE=1", "SAM3D_IMAGE_GATHER=1", "SAM3D_BF16_PRECISE_PREFIX=1", "GGML_VK_FUSE_BF16_ROUND=1", "GGML_VK_FUSE_BF16_BINARY=1", "GGML_VK_BF16_BINARY_LINEAR=1", "GGML_VK_BF16_MATMUL_TILE=small", "SAM3D_BATCHED_TRANSFERS=1", "SAM3D_SIMD_SKINNING=1", "GGML_VK_F32_NARROW_MATMUL=1", "GGML_VK_F32_NARROW_TILE=tiny32")
	}
	return append(out, "GGML_VK_DISABLE_COOPMAT=1", "GGML_VK_DISABLE_COOPMAT2=1")
}

func workerRequest(input, output string) []byte {
	var b bytes.Buffer
	for _, path := range []string{input, output} {
		_ = binary.Write(&b, binary.LittleEndian, uint32(len(path)))
		b.WriteString(path)
	}
	return b.Bytes()
}

func (a *app) runBodySequence(ctx context.Context, id string, frames [][]string, outputDir string) ([]string, error) {
	c := a.cfg
	args := []string{"--worker", c.bodyModule, c.backend, strconv.Itoa(c.device), c.deviceName, c.backbone, c.branch, c.mhr, strconv.Itoa(c.threads), "--gem-features"}
	if c.bf16 {
		args = append(args, "--bf16")
	}
	cmd := exec.CommandContext(ctx, c.bodyRunner, args...)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Env = c.bodyEnvironment(os.Environ())
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	logFile, err := os.OpenFile(filepath.Join(a.dir(id), "inference.log"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, err
	}
	defer logFile.Close()
	cmd.Stderr = logFile
	if err = cmd.Start(); err != nil {
		return nil, fmt.Errorf("start SAM3D Body worker: %w", err)
	}
	done := make(chan error, 1)
	scanner := bufio.NewScanner(stdout)
	go func() { done <- cmd.Wait() }()
	exited := false
	defer func() {
		if !exited {
			_ = stdin.Close()
			select {
			case <-done:
			default:
				_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
				<-done
			}
		}
	}()
	next := func(expected string) error {
		for {
			scanned := make(chan bool, 1)
			go func() { scanned <- scanner.Scan() }()
			select {
			case <-ctx.Done():
				return ctx.Err()
			case ok := <-scanned:
				if !ok {
					return fmt.Errorf("SAM3D Body worker exited before %s", expected)
				}
				line := scanner.Text()
				_, _ = fmt.Fprintln(logFile, line)
				if line == expected {
					return nil
				}
				if strings.HasPrefix(line, "TIMING ") {
					continue
				}
				return fmt.Errorf("unexpected SAM3D Body reply %q", line)
			}
		}
	}
	if err = next("READY"); err != nil {
		return nil, err
	}
	results := make([]string, len(frames))
	for i, frame := range frames {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}
		results[i] = filepath.Join(outputDir, fmt.Sprintf("%06d.bin", i))
		request := workerRequest(frame[0], results[i])
		if _, err = io.Copy(stdin, bytes.NewReader(request)); err != nil {
			return nil, err
		}
		if err = next("DONE"); err != nil {
			return nil, err
		}
		a.stage(id, fmt.Sprintf("Extracting SAM3D Body pose features (%d/%d)", i+1, len(frames)))
	}
	_ = stdin.Close()
	if err = <-done; err != nil {
		return nil, fmt.Errorf("SAM3D Body worker: %w", err)
	}
	exited = true
	return results, nil
}
