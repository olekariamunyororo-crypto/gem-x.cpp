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

func TestSequenceManifestPreservesObservationBox(t *testing.T) {
	path := filepath.Join(t.TempDir(), "sequence.manifest")
	if err := writeSequenceManifest(path, [][]string{{"image", "body"}}, [][4]float32{{10, 20, 40, 60}}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil || string(b[:8]) != "GEMSEQ02" || binary.LittleEndian.Uint32(b[8:12]) != 1 {
		t.Fatalf("invalid sequence manifest: %v", err)
	}
	r := bytes.NewReader(b[12:])
	for _, want := range []string{"image", "body"} {
		var count uint32
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			t.Fatal(err)
		}
		value := make([]byte, count)
		if _, err := r.Read(value); err != nil || string(value) != want {
			t.Fatalf("manifest path: %q, %v", value, err)
		}
	}
	var box [3]float32
	if err := binary.Read(r, binary.LittleEndian, &box); err != nil || box != [3]float32{25, 40, 48} || r.Len() != 0 {
		t.Fatalf("manifest observation box: %v, %v", box, err)
	}
}

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
	wantFocal := float32(20)
	for _, offset := range []int{36, 40} {
		got := math.Float32frombits(binary.LittleEndian.Uint32(packed[offset:]))
		if got != wantFocal {
			t.Fatalf("focal = %g, want %g", got, wantFocal)
		}
	}
	box := bodyBox([4]float32{10, 20, 110, 220})
	wantBox := [4]float32{-60, 0, 180, 240}
	for i := range box {
		if math.Abs(float64(box[i]-wantBox[i])) > 1e-4 {
			t.Fatalf("body crop = %v", box)
		}
	}
	wide := bodyBox([4]float32{10, 20, 210, 120})
	wantWide := [4]float32{-50, -90, 270, 230}
	for i := range wide {
		if math.Abs(float64(wide[i]-wantWide[i])) > 1e-4 {
			t.Fatalf("wide body crop = %v", wide)
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
