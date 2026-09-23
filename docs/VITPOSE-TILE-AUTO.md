# Automatic ViTPose tile selection (2026-09-22/23)

This document describes the expanded heuristics.

## Motivation

The original tile tuning delivered an 11.8 % reduction on the RTX 5070 Ti for the live batch-2 shape by forcing 32×32 tiles on the gate/up and attention-output projections. The selection was hard-coded to a single device name.

The new logic:

- Matches a short allow-list of common high-end consumer NVIDIA GPUs by substring.
- Falls back gracefully on unknown devices (upstream GGML selection is kept).
- Still restricts the change to the validated live (N=197, batch=2) and offline (N=394) shapes and to strict F32.
- Adds an opt-in QKV tile path (`GEMX_VITPOSE_QKV_TILES=1`) so register-pressure experiments can be run without touching the default path.
- Preserves the original split-K behaviour for the down projection.

## Environment variables

| Variable | Default | Meaning |
|----------|---------|--------|
| `GEMX_VITPOSE_TILES` | on | Master switch (set to `0` to disable all custom tiles) |
| `GEMX_VITPOSE_DOWN_TILES` | on | Allow the 128×32 down-projection override |
| `GEMX_VITPOSE_QKV_TILES` | off | Opt-in QKV lower-register tile (experimental) |
| `GEMX_VITPOSE_TILES_TRACE` | off | Log the first activation of each tile class |
| `GEMX_VITPOSE_RECT_GROUP` | `up` | Legacy rectangular experiment group |

## Expected impact

On devices that already had the 32×32 path (5070 Ti) the behaviour is unchanged. On the newly allowed devices the same relative gain that was measured on the 5070 Ti is expected for the gate/up and attention-output projections.

QKV remains on the upstream tile by default. The opt-in path lets further micro-architecture work be evaluated safely.

## Rollback

```bash
export GEMX_VITPOSE_TILES=0
# or rebuild with -DGEMX_VITPOSE_TILES=OFF
```
