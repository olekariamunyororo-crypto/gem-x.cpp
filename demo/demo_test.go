package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestPackedFrameAndBodyCrop(t *testing.T) {
	im := image.NewRGBA(image.Rect(0, 0, 16, 12))
	im.SetRGBA(3, 4, color.RGBA{R: 11, G: 22, B: 33, A: 255})
	packed := packImage(im)
	if len(packed) != packedHeader+16*12*3 || string(packed[:8]) != "S3DIMG01" {
		t.Fatal("wrong packed frame")
	}
	for offset, want := range map[int]uint32{8: 16, 12: 12, 16: 48} {
		if got := binary.LittleEndian.Uint32(packed[offset:]); got != want {
			t.Fatalf("header[%d]=%d, want %d", offset, got, want)
		}
	}
	box := bodyBox([4]float32{10, 20, 110, 220})
	wantBox := [4]float32{-60, 0, 180, 240}
	for i := range box {
		if math.Abs(float64(box[i]-wantBox[i])) > 1e-4 {
			t.Fatalf("body crop = %v", box)
		}
	}
	if err := patchPackedBox(packed, box); err != nil {
		t.Fatal(err)
	}
	for i, want := range box {
		if got := math.Float32frombits(binary.LittleEndian.Uint32(packed[20+i*4:])); got != want {
			t.Fatalf("box[%d]=%g, want %g", i, got, want)
		}
	}
}

func TestManifestAndBoxWireFormats(t *testing.T) {
	dir := t.TempDir()
	manifest := filepath.Join(dir, "images.manifest")
	rows := [][]string{{"/tmp/a.input"}, {"/tmp/b.input"}}
	if err := writePathManifest(manifest, "GEMIMGS1", rows); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(manifest)
	if err != nil || string(b[:8]) != "GEMIMGS1" || binary.LittleEndian.Uint32(b[8:]) != 2 {
		t.Fatal("wrong manifest header", err)
	}
	var boxes bytes.Buffer
	boxes.WriteString("GEMBOX01")
	_ = binary.Write(&boxes, binary.LittleEndian, uint32(2))
	want := [][4]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	_ = binary.Write(&boxes, binary.LittleEndian, want)
	path := filepath.Join(dir, "boxes.bin")
	if err = os.WriteFile(path, boxes.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	got, err := readBoxes(path, 2)
	if err != nil || len(got) != 2 || got[1] != want[1] {
		t.Fatalf("boxes = %v, %v", got, err)
	}
}
