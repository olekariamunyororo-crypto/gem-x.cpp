# Aggressive QKV Register Micro-Tile Experiment (2026-09-22)

## Goal
Reduce the 16.7 % dispatch-stall share and 255 registers/thread observed on the default Vulkan QKV (M=3840, N=394, K=1280) while preserving FP32 accumulation order and byte-identical (or near-identical under a documented tolerance) heatmaps.

## Proposed experiment matrix

| Variant | Output tile | Target regs/thread | Shared/block | Notes |
|---------|-------------|--------------------|--------------|-------|
| Baseline (upstream) | 128×128 | 255 | 48 KiB | Current production |
| A – 128×64 (existing) | 128×64 | ~183 | ~26 KiB | Already measured, slower |
| B – 64×64 micro | 64×64 | ≤128 | ≤24 KiB | Goal: 3+ resident blocks |
| C – 64×32 micro | 64×32 | ≤96 | ≤16 KiB | Aggressive, higher grid |
| D – 32×32 micro | 32×32 | ≤64 | ≤12 KiB | Maximum occupancy experiment |

Enable with:
```bash
GEMX_VITPOSE_QKV_TILES=1 GEMX_VITPOSE_QKV_LAYOUT=64x64-micro
# or 64x32-micro / 32x32-micro
```

## Acceptance criteria
- ≥5 % reduction in isolated QKV time **or** ≥3 % reduction in full ViTPose time
- Byte-identical heatmaps on the 72-frame real-video crop set (or documented max abs error < 1e-5)
- No increase in register spills that re-introduce long scoreboard stalls

## Status
Layouts are registered in `vulkan_vitpose_experiments.inc`. Measure on target hardware before promoting any variant to default.
