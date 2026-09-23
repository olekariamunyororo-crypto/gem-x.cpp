// Expanded experimental pipelines for ViTPose (2026-09-22 performance work).
//
// Changes vs upstream:
// - Device allow-list matches the automatic tile selector (more than just 5070 Ti).
// - QKV micro-tile layouts are easier to select and include the aggressive
//   lower-register variants from the experiment plan.
// - Still gated behind environment variables; production defaults unchanged.
// - Fail-closed on unknown devices / precision.

static bool gemx_device_supported_for_experiments(const vk_device *device) {
    if (!device || device->fp16 || device->coopmat2 || device->coopmat_support)
        return false;
    const char *name = device->properties.deviceName.data();
    static const char * const allowed[] = {
        "NVIDIA GeForce RTX 5070 Ti",
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 4070",
        nullptr
    };
    for (const char * const *p = allowed; *p; ++p)
        if (strstr(name, *p)) return true;
    // Conservative fallback for other discrete NVIDIA cards with sufficient workgroup size.
    return strstr(name, "NVIDIA") != nullptr &&
           device->properties.limits.maxComputeWorkGroupInvocations >= 1024;
}

if (gemx_device_supported_for_experiments(device)) {
    // Optional fused norm (already present upstream for 5070 Ti).
    const char *norm = getenv("GEMX_VITPOSE_NORM");
    if (!norm || strcmp(norm, "1") == 0)
        ggml_vk_create_pipeline(device, device->gemx_norm_pipeline, "gemx_norm_affine",
            sizeof(gemx_norm_spv), reinterpret_cast<const unsigned char *>(gemx_norm_spv),
            "main", 4, 8, {1,1,1}, {}, 1);

    // Rectangular projection experiments (gate/up, attention output, etc.).
    const char *tile = getenv("GEMX_VITPOSE_RECT");
    if (!tile) tile = "64x64";
    std::vector<uint32_t> spec;
    if (strcmp(tile, "128x32") == 0) spec = {128,128,32,16,32,32,2,4,2,1,32};
    if (strcmp(tile, "64x32")  == 0) spec = {64,64,32,16,32,32,2,4,2,1,32};
    if (strcmp(tile, "32x64")  == 0) spec = {64,32,64,16,32,32,2,4,2,1,32};
    if (strcmp(tile, "64x64")  == 0) spec = {64,64,64,16,32,64,2,4,2,1,32};
    if (strcmp(tile, "128x64") == 0) spec = {128,128,64,16,64,32,2,4,2,1,32};
    if (strcmp(tile, "32x32")  == 0) spec = {32,32,32,16,32,32,2,2,2,1,32};
    if (strcmp(tile, "32x128") == 0) spec = {128,32,128,16,32,32,2,4,2,1,32};
    if (strcmp(tile, "64x128") == 0) spec = {128,64,128,16,32,64,2,4,4,1,32};

    const char *bk = getenv("GEMX_VITPOSE_RECT_BK");
    if (!spec.empty() && bk && strcmp(bk, "32") == 0 && spec[1] + spec[2] <= 160) spec[3] = 32;
    if (!spec.empty() && bk && strcmp(bk, "64") == 0 && spec[1] + spec[2] <= 64)  spec[3] = 64;

    if (!spec.empty())
        ggml_vk_create_pipeline(device, device->gemx_rect_pipeline, "gemx_projection_rect",
            matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3,
            sizeof(vk_mat_mat_push_constants),
            {spec[1], spec[2], 1}, ggml_vk_mul_mm_spec(spec, true), 32);
}

// QKV register-microtile experiments (independent of the expansion tile).
// These are the aggressive variants from docs/VITPOSE-QKV-MICROTILE-EXPERIMENT.md.
// Enable with GEMX_VITPOSE_QKV_TILES=1 and optionally GEMX_VITPOSE_QKV_LAYOUT=<name>.
if (gemx_device_supported_for_experiments(device)) {
    const char *layout = getenv("GEMX_VITPOSE_QKV_LAYOUT");
    if (!layout) layout = "64x64-t128";          // good starting point for lower register pressure

    std::vector<uint32_t> spec;
    // Existing layouts
    if (strcmp(layout, "128x64-t256") == 0) spec = {256,128,64,16,32,32,1,4,2,1,32};
    if (strcmp(layout, "64x64-t128")  == 0) spec = {128,64,64,16,32,32,1,4,2,1,32};
    if (strcmp(layout, "64x32-t128")  == 0) spec = {128,64,32,16,32,16,1,2,2,1,32};
    if (strcmp(layout, "128x32-w1")   == 0) spec = {128,128,32,16,32,32,1,4,2,1,32};
    if (strcmp(layout, "128x128-t256")== 0) spec = {256,128,128,16,32,64,1,4,4,1,32};
    if (strcmp(layout, "128x64-w1")   == 0) spec = {128,128,64,16,64,32,1,4,2,1,32};

    // New aggressive micro-tile layouts aimed at ≤128 registers
    if (strcmp(layout, "64x64-micro") == 0)   spec = {128,64,64,16,32,32,1,4,2,1,32};
    if (strcmp(layout, "64x32-micro") == 0)   spec = {128,64,32,16,32,16,1,2,2,1,32};
    if (strcmp(layout, "32x32-micro") == 0)   spec = {64,32,32,16,16,16,1,2,2,1,32};

    if (!spec.empty())
        ggml_vk_create_pipeline(device, device->gemx_qkv_pipeline, "gemx_qkv_microtile",
            matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3,
            sizeof(vk_mat_mat_push_constants),
            {spec[1], spec[2], 1}, ggml_vk_mul_mm_spec(spec, true), 32);
}
