# F32 FlashAttention experiment — 2026-09-20

GGML Vulkan FlashAttention works for ViTPose, but does not improve measured end-to-end performance on the RTX 5070 Ti in the strict-FP32 configuration. It remains **off by default**.

| Case | Existing attention | FlashAttention |
|---|---:|---:|
| Live batch two, 200 iterations | 49.39 ms | 49.80 ms |
| Live repeat, 200 iterations | 49.71 ms | 50.13 ms |
| Offline batch eight, 50 iterations | 203.10 ms | 207.15 ms |

Read exact figures from the [machine-readable report](vitpose-flash-2026-09-20.json); timings include transfers, exclude preprocessing, and follow ten warmups. Live runs use identical normalized pinned crop/flip inputs. Offline timing uses the benchmark's deterministic synthetic input. All compilation and runtime use at most eight CPU cores. GPU clocks are unmodified; normal clock and thermal variation remains. No useful speedup was demonstrated.

The fused attention operations total 3.89 ms across 32 blocks, versus approximately 3.99 ms for the preceding matched-input attention matmuls plus softmax. This does not affect the QKV or feed-forward projections. F16 and cooperative matrices remain disabled; the experiment does not use the faster reduced-precision Tensor Core paths.

Enable with `GEMX_VITPOSE_FLASH_ATTN=1`, alongside the existing strict-F32 and accepted optimization flags. The graph retains separate upstream Q/K scaling, uses F32 Q/K/V, requests F32 accumulation, and adapts the fused operation's already-permuted result layout. FlashAttention changes reduction order, so byte equality is not expected.

All 72 saved real-video crops were compared at batch sizes two and eight: zero changed raw peaks (11,088 per batch-size test), zero changed flip-averaged, quarter-pixel decoded coordinates (5,544 per test), maximum heatmap difference 8.73e-6 and mean 1.04e-8. Maximum decoded confidence difference is 4.29e-6. This is ViTPose parity, not a new downstream motion replay. The ordinary disabled path remains byte-identical to the previous fixture baseline. Five CTest contracts passed.

Reproduce timings with `taskset -c 0-7 python scripts/profile_vitpose.py`, passing the flag above through the environment. `check_vitpose_crops.py --save-heatmaps DIR` saves real-crop results; `compare_vitpose_heatmaps.py BASELINE CANDIDATE --output REPORT` compares numerical heatmaps and native flip/quarter-pixel decoding. Local raw artifacts are under `generated/vitpose-flash-20260920/`.
