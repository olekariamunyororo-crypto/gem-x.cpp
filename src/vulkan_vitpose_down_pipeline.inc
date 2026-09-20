// Specialize the existing full-F32 shader; compilation remains lazy.
// See docs/VITPOSE-DOWN-PROJECTION.md. No changes to dot-product arithmetic.
if (!device->fp16 && !device->coopmat2 && !device->coopmat_support &&
    strcmp(device->properties.deviceName.data(), "NVIDIA GeForce RTX 5070 Ti") == 0) {
    // Threads, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP.
    const std::vector<uint32_t> spec = {128, 128, 32, 16, 32, 32, 2, 4, 2, 1, 32};
    ggml_vk_create_pipeline(device, device->gemx_vitpose_down_pipeline, "gemx_vitpose_down_f32_128x32",
        matmul_f32_f32_fp32_len, matmul_f32_f32_fp32_data, "main", 3, sizeof(vk_mat_mat_push_constants),
        {128, 32, 1}, ggml_vk_mul_mm_spec(spec, true), 32);
}
