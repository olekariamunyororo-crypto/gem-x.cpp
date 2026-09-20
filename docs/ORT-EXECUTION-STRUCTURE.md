# CUDA reference execution structure — 2026-09-20

The independent ONNX Runtime CUDA benchmark uses the same exported networks.
It changes execution structure and kernel implementations; enabling TF32 also
changes arithmetic precision. No alternative backend is integrated.

Previously measured uninstrumented ViTPose times are 54.508 ms native,
40.866 ms CUDA with TF32 disabled, and 24.564 ms with TF32 enabled. YOLOX
reference points are 55.463, 29.829 and 17.395 ms respectively. Timing scopes
and fixture accuracy are documented in [the benchmark report](LIVE-DETECTION-CADENCE.md).
In particular, TF32 changed two of 154 raw heatmap peaks on the fixture.

## Actual operation trace

Captured ORT 1.23.2 with TF32 disabled, two warm-ups and three timed runs,
sequentially for each model, with CPU affinity 0–7. The benchmark script now
accepts `--profile --warmups 2 --iterations 3`. These captures establish
executed operators and tensor shapes, not uninstrumented performance.

The final ViTPose run contains 1,816 ORT node events: 1,649 assigned to CUDA
and 167 to CPU. It includes 192 MatMul, 32 FusedMatMul, 64 LayerNormalization,
one SkipLayerNormalization, 32 QuickGelu and 32 Softmax operations. Attention
still uses separate score multiplication, softmax and value multiplication;
there is no fused Attention operator in this capture. FusedMatMul names identify
MatMulScaleFusion. Many additions, multiplications and layout operations remain.

YOLOX contains 427 node events, of which 423 are assigned to CUDA and four to
CPU. It has 155 Conv and 146 QuickGelu operations.

These are **host operation events, not GPU kernel counts or GPU durations**.
A node can launch multiple kernels or perform bookkeeping. Do not compare
these counts directly with native graph operations or sum their durations to
attribute GPU time. The CPU assignments do not establish CPU stalls.

## Concrete differences

### Projection batching

The trace shows projection input `[2,197,1280]` and shared two-dimensional
weights, including `[1280,5120]` for the feed-forward expansion. ORT's matching
[MatMul helper](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/core/providers/cpu/math/matmul_helper.h)
flattens those leading dimensions to 394 rows. Its
[CUDA MatMul implementation](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/core/providers/cuda/math/matmul.cc)
then selects a single cuBLAS GEMM call. This is inferred from the captured
shapes and the exact-version source, not a captured cuBLAS API trace.

Our profiled Vulkan projections retain two matrices of 197 tokens each.
Combining them into 394 tokens preserves the logical dot products while
changing tiling opportunities and tile-edge waste. It does not halve the
arithmetic. Weight reuse and speed improvement depend on the selected kernel;
neither has been measured for a flattened Vulkan implementation.

### Fusions and convolution implementation

ORT executes normalization with scale and bias as LayerNormalization; the
native graph expresses normalization, multiplication and addition separately
in `src/vitpose.cpp`. The trace also confirms activation and attention-scale
fusions, but not wholesale transformer-block fusion.

Native YOLOX explicitly materializes F32 im2col, runs a matrix multiplication,
copies the resulting layout, and adds bias for each convolution
(`src/yolox.cpp`). ORT delegates convolution to cuDNN with exhaustive algorithm
search and maximum workspace enabled. This permits a different convolution
implementation rather than requiring our explicit im2col pipeline. The exact
cuDNN algorithm and internal memory traffic were not captured.

### Precision and disabled features

[ORT's CUDA provider documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
explains that TF32 permits reduced-precision inputs with FP32 accumulation
for supported operations. The 40.866 → 24.564 ms ViTPose improvement is from
enabling that option within the same runtime, not changing the architecture.
This is not a strict-parity optimization. Tensor Core instruction utilization
was not measured in these host traces.

Recorded options have CUDA Graph replay, preferred NHWC layout and tunable
operators disabled. The benchmark uses CUDAExecutionProvider, not TensorRT.
Those features therefore do not explain these measurements.

## What follows from this

Flattening crop and flip into one projection matrix is a concrete Vulkan
experiment that does not require a new backend. It needs benchmarking and
parity checks: a changed shape may select different reduction algorithms.
For YOLOX, avoiding explicit convolution intermediates offers a larger
structural direction but requires more implementation work. Neither is a
demonstrated speedup yet.

GPU kernel/API tracing would be needed to quantify cuBLAS/cuDNN contributions,
their selected kernels and launch gaps. These results do not establish CUDA
occupancy, instruction stalls or bandwidth utilization.

The committed [trace summary](ort-execution-structure-2026-09-20.json) records
operator counts, representative shapes and provider options. Raw captures are
under `generated/ort-structure-20260920/`. The profiling script passed syntax
checks and successfully captured both models; application inference is unchanged.
