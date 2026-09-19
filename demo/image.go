package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"math"
)

const packedHeader = 52

func decodeImage(data []byte) (image.Image, string, error) {
	c, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil || (format != "jpeg" && format != "png") {
		return nil, "", fmt.Errorf("frame must be a JPEG or PNG")
	}
	if c.Width < 8 || c.Height < 8 || c.Width > 4096 || c.Height > 4096 || int64(c.Width)*int64(c.Height) > 1_000_000 {
		return nil, "", fmt.Errorf("frame must be 8–4096 pixels per side and at most 1 megapixel")
	}
	im, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, "", fmt.Errorf("decode frame: %w", err)
	}
	return im, format, nil
}

func packImage(im image.Image) []byte {
	b := im.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]byte, packedHeader+w*h*3)
	copy(out, "S3DIMG01")
	binary.LittleEndian.PutUint32(out[8:], uint32(w))
	binary.LittleEndian.PutUint32(out[12:], uint32(h))
	binary.LittleEndian.PutUint32(out[16:], uint32(w*3))
	box := [4]float32{0, 0, float32(w), float32(h)}
	// SAM 3D Body's prepare_batch uses the image diagonal when no calibrated
	// camera is supplied. GEM-X independently uses max(width,height) below;
	// these two published defaults are deliberately different.
	focal := float32(math.Hypot(float64(w), float64(h)))
	camera := [4]float32{focal, focal, float32(w) / 2, float32(h) / 2}
	for i, v := range box {
		binary.LittleEndian.PutUint32(out[20+i*4:], math.Float32bits(v))
	}
	for i, v := range camera {
		binary.LittleEndian.PutUint32(out[36+i*4:], math.Float32bits(v))
	}
	dst, at := out[packedHeader:], 0
	switch p := im.(type) {
	case *image.YCbCr:
		for y := b.Min.Y; y < b.Max.Y; y++ {
			yi := p.YOffset(b.Min.X, y)
			for x := b.Min.X; x < b.Max.X; x++ {
				ci := p.COffset(x, y)
				r, g, blue := color.YCbCrToRGB(p.Y[yi], p.Cb[ci], p.Cr[ci])
				dst[at], dst[at+1], dst[at+2] = r, g, blue
				yi++
				at += 3
			}
		}
	default:
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				r, g, blue, _ := im.At(x, y).RGBA()
				dst[at], dst[at+1], dst[at+2] = byte(r>>8), byte(g>>8), byte(blue>>8)
				at += 3
			}
		}
	}
	return out
}

func patchPackedBox(data []byte, box [4]float32) error {
	if len(data) < packedHeader || string(data[:8]) != "S3DIMG01" {
		return fmt.Errorf("invalid packed frame")
	}
	for i, v := range box {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return fmt.Errorf("detector returned a nonfinite box")
		}
		binary.LittleEndian.PutUint32(data[20+i*4:], math.Float32bits(v))
	}
	return nil
}

func bodyBox(box [4]float32) [4]float32 {
	xys := observationBox(box)
	cx, cy, size := xys[0], xys[1], xys[2]
	return [4]float32{cx - size*.5, cy - size*.5, cx + size*.5, cy + size*.5}
}

func observationBox(box [4]float32) [3]float32 {
	w, h := box[2]-box[0], box[3]-box[1]
	// This is GEM-X get_bbx_xys_from_xyxy(base_enlarge=1.2): fit the detector
	// box to the published 192:256 prior, then enlarge it and make it square.
	// SAM 3D Body applies its own preprocessing transform to this supplied box.
	size := max(h, w/.75) * 1.2
	cx, cy := (box[0]+box[2])*.5, (box[1]+box[3])*.5
	return [3]float32{cx, cy, size}
}
