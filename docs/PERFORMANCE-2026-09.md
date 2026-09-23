# Performance improvements — 2026-09-22/23

## Summary
- Expanded automatic ViTPose tile selection to more NVIDIA consumer GPUs.
- Safer opt-in QKV micro-tile path with aggressive lower-register layouts.
- Adaptive fence-wait infrastructure (GEMX_FENCE_POLL_SLEEP).
- Extended experiment + parity tooling.

See APPLY.md and the individual docs for details and runtime controls.
