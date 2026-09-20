// Included after GGML selects a matmul pipeline. See docs/VITPOSE-TILE-TUNING.md.
// Change output tiling only for the validated device, precision and live shapes.
// Keep QKV on upstream selection; preserve the down projection's split-K.
vk_pipeline gemx_vitpose_split_pipeline;
{
    static const bool gemx_vitpose_tiles = [] {
        const char *value = getenv("GEMX_VITPOSE_TILES");
        return !value || strcmp(value, "0") != 0;
    }();
    if (gemx_vitpose_tiles && strcmp(ctx->device->properties.deviceName.data(), "NVIDIA GeForce RTX 5070 Ti") == 0 &&
        !ctx->device->fp16 && !ctx->device->coopmat2 && !ctx->device->coopmat_support &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
        ne11 == 197 && ne12 * ne13 == 2) {
        unsigned bit = 0;
        if (ne10 == 1280 && (ne01 == 5120 || ne01 == 1280) && ctx->device->mul_mat_s[GGML_TYPE_F32]) {
            pipeline = aligned ? mmp->a_s : mmp->s;
            bit = ne01 == 5120 ? 1 : 2;
        }
        static const bool down_tiles = [] {
            const char *value = getenv("GEMX_VITPOSE_DOWN_TILES");
            return !value || strcmp(value, "0") != 0;
        }();
        if (down_tiles && aligned && ne10 == 5120 && ne01 == 1280 && ctx->device->gemx_vitpose_down_pipeline) {
            gemx_vitpose_split_pipeline = pipeline;
            pipeline = ctx->device->gemx_vitpose_down_pipeline;
            bit = 4;
        }
        static const bool trace = getenv("GEMX_VITPOSE_TILES_TRACE") != nullptr;
        static thread_local unsigned traced = 0;
        if (trace && bit && !(traced & bit)) {
            GGML_LOG_INFO("gemx_vitpose_tiles: m=%llu n=197 k=%llu batch=2 tile=%s%s\n",
                          static_cast<unsigned long long>(ne01), static_cast<unsigned long long>(ne10),
                          bit == 4 ? "128x32" : "32", bit == 4 ? " original-split-k" : "");
            traced |= bit;
        }
    }
}
