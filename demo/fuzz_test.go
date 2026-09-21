package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"image/png"
	"math"
	"testing"
)

func FuzzImage(f *testing.F) {
	var seed bytes.Buffer
	_ = png.Encode(&seed, image.NewRGBA(image.Rect(0, 0, 8, 8)))
	f.Add(seed.Bytes())
	var jpegSeed bytes.Buffer
	_ = jpeg.Encode(&jpegSeed, image.NewRGBA(image.Rect(0, 0, 8, 8)), nil)
	f.Add(jpegSeed.Bytes())
	f.Add([]byte("not an image"))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > maxFrameBytes {
			return
		}
		im, _, err := decodeImage(data)
		if err != nil {
			return
		}
		packed := packImage(im)
		if len(packed) != packedHeader+im.Bounds().Dx()*im.Bounds().Dy()*3 {
			t.Fatal("packed length")
		}
		if string(packed[:8]) != "S3DIMG01" {
			t.Fatal("packed magic")
		}
	})
}

func FuzzPackedBox(f *testing.F) {
	f.Add(make([]byte, 52), uint32(0), uint32(0), uint32(8), uint32(8))
	f.Add(packImage(image.NewRGBA(image.Rect(0, 0, 8, 8))), uint32(0), uint32(0), math.Float32bits(8), math.Float32bits(8))
	f.Fuzz(func(t *testing.T, data []byte, a, b, c, d uint32) {
		before := append([]byte(nil), data...)
		box := [4]float32{math.Float32frombits(a), math.Float32frombits(b), math.Float32frombits(c), math.Float32frombits(d)}
		if patchPackedBox(data, box) != nil {
			return
		}
		if !bytes.Equal(data[:20], before[:20]) || !bytes.Equal(data[36:], before[36:]) {
			t.Fatal("box patch corrupted frame")
		}
		for i, bits := range []uint32{a, b, c, d} {
			if binary.LittleEndian.Uint32(data[20+4*i:]) != bits {
				t.Fatal("box roundtrip")
			}
		}
	})
}

func FuzzCreateJSON(f *testing.F) {
	f.Add([]byte(`{"name":"clip","width":8,"height":8,"frames":2,"fps":10}`))
	f.Add([]byte(`{"width":1e100,"frames":-1}`))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 8192 {
			return
		}
		// A zero job limit exercises parsing/validation without filesystem writes.
		a := &app{cfg: config{maxJobs: 0}, jobs: map[string]*job{}}
		w := liveRequest(a, "POST", "/api/jobs", data)
		if w.Code != 400 && w.Code != 507 {
			t.Fatalf("unexpected status: %d", w.Code)
		}
	})
}

func FuzzDetectorBoxes(f *testing.F) {
	seed := make([]byte, 28)
	copy(seed, "GEMBOX01")
	binary.LittleEndian.PutUint32(seed[8:], 1)
	f.Add(seed, 1)
	f.Add([]byte{}, -1)
	f.Fuzz(func(t *testing.T, data []byte, count int) {
		boxes, err := parseBoxes(data, count)
		if err != nil {
			return
		}
		if len(boxes) != count || count < 1 || count > maxFrames {
			t.Fatal("invalid accepted count")
		}
		for _, box := range boxes {
			for _, value := range box {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatal("invalid accepted coordinate")
				}
			}
		}
	})
}
