# ViTPose defaults — 2026-09-21

The validated optimization set is now enabled automatically. No `GEMX_VITPOSE_*` environment variables are needed for the normal demo or native pipeline.

| Optimization | Default | Override |
|---|---|---|
| Combine crop/flip token rows in projection matmuls | On, contiguous batch-two inputs | `GEMX_VITPOSE_FLATTEN=0` |
| Fused layer normalization/scale/bias | On for validated Vulkan shape | `GEMX_VITPOSE_NORM=0` |
| Fused SwiGLU | On | `GEMX_VITPOSE_SWIGLU=0` |
| Expansion projection tile | 64×64 | `GEMX_VITPOSE_RECT=0` |
| QKV register/workgroup layout | 64×64, 128 threads | `GEMX_VITPOSE_QKV_LAYOUT=0` |
| FlashAttention | Off | `GEMX_VITPOSE_FLASH_ATTN=1` for experiments |
| Packed gate/up weights | Off | `GEMX_VITPOSE_PACK_FFN=1` for experiments |

The custom Vulkan kernels retain their RTX 5070 Ti, F32 and shape checks. F16 and cooperative matrices must be disabled for that path; the demo already enforces this precision configuration for ViTPose. This does not change backend-wide precision defaults or enable TF32. Other shapes retain their appropriate existing kernels. Batch-eight projection layout is unchanged; SwiGLU is also fused there.

`GEMX_VITPOSE_TILES=0` disables all tile overrides, including expansion and QKV. It does not disable flattening, normalization or SwiGLU; use their individual controls for an unfused diagnostic. Explicit alternate tile/layout values still work. Historical experiment scripts explicitly set their old baseline controls rather than accidentally measuring the new defaults.

These defaults supersede earlier experiment documents describing these optimizations as opt-in. See [projection measurements](VITPOSE-PROJECTION-EXPERIMENTS.md) and [FlashAttention results](VITPOSE-FLASH-ATTENTION.md). The best prior measured latency was about 49.1 ms, compared with CUDA's 41.2 ms; this change enables that configuration automatically, not a new speedup beyond it.

The existing devroute Go service launches the rebuilt native pipeline/backend for each new live session or native stage. No Go/UI change or service restart is required when no native worker remains active.

Validation with **no ViTPose environment overrides**: 200 timed batch-two inferences averaged **48.92 ms**. All 72 saved real-video crops produced byte-identical heatmaps at batch sizes two and eight. The pinned fixture matched both the preceding explicit winning configuration and an all-opt-outs run. All five CTest contracts passed. The demo has no inherited ViTPose overrides and devroute returned HTTP 200. [Measurement report](vitpose-defaults-2026-09-21.json).
