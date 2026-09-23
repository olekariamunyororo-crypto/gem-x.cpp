# How to apply / use the 2026-09 performance improvements

## Runtime controls (all optional)

| Variable | Default | Effect |
|----------|---------|--------|
| `GEMX_VITPOSE_TILES` | on | Master switch for automatic tile selection |
| `GEMX_VITPOSE_QKV_TILES` | off | Enable the experimental QKV micro-tile pipeline |
| `GEMX_VITPOSE_QKV_LAYOUT` | `64x64-t128` | Choose layout (`64x64-micro`, `64x32-micro`, `32x32-micro`, …) |
| `GEMX_VITPOSE_RECT` | `64x64` | Rectangular projection experiment |
| `GEMX_FENCE_POLL_SLEEP` | off | Adaptive fence wait (reduces CPU spin) |
| `GEMX_VITPOSE_TILES_TRACE` | off | Log which tiles / pipelines are selected |

## Recommended first experiments

```bash
# Safe automatic tiles (more GPUs)
./demo/gemx-demo --threads 8

# Aggressive QKV micro-tile (measure carefully)
GEMX_VITPOSE_QKV_TILES=1 GEMX_VITPOSE_QKV_LAYOUT=64x64-micro \
  ./demo/gemx-demo --threads 8

# Lower CPU spin during GPU wait
GEMX_FENCE_POLL_SLEEP=1 ./demo/gemx-demo --threads 8
```

All changes are reversible by environment variable.
