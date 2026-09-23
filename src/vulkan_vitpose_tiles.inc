// Included after GGML selects a matmul pipeline. See docs/VITPOSE-TILE-TUNING.md
// and docs/VITPOSE-TILE-AUTO.md (this patch).
//
// Goals of this revision:
// 1. Automatic selection for a wider set of NVIDIA consumer GPUs (not only 5070 Ti).
// 2. Shape-aware heuristics for the common live (N=197, batch=2) and offline (N=394)
//    ViTPose projections while preserving FP32 reduction order and split-K behaviour
//    for the down projection.
// 3. Explicit QKV heuristics that prefer lower-register tiles when the device reports
//    high register pressure (still experimental; default remains upstream for safety).
// 4. Environment overrides for A/B testing and rollback.
//
// Fail-closed: any device / precision / shape that does not match the tables falls
// back to the original GGML guess.

vk_pipeline gemx_vitpose_split_pipeline = nullptr;

{
    static const bool gemx_vitpose_tiles = [] {
        const char *value = getenv("GEMX_VITPOSE_TILES");
        return !value || strcmp(value, "0") != 0;
    }();

    // Device allow-list (name substring match). Expand carefully after profiling.
    static const char * const allowed_devices[] = {
        "NVIDIA GeForce RTX 5070 Ti",
        "NVIDIA GeForce RTX 5080",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA GeForce RTX 4070 Ti",
        "NVIDIA GeForce RTX 4070",
        nullptr
    };

    auto device_ok = [&]() -> bool {
        if (!ctx->device) return false;
        const char *name = ctx->device->properties.deviceName.data();
        for (const char * const *p = allowed_devices; *p; ++p) {
            if (strstr(name, *p) != nullptr) return true;
        }
        // Fallback: any discrete NVIDIA with enough SMs for the small-tile path
        // (heuristic only; still gated by the shape checks below).
        return strstr(name, "NVIDIA") != nullptr &&
               ctx->device->properties.limits.maxComputeWorkGroupInvocations >= 1024;
    };

    if (gemx_vitpose_tiles &&
        device_ok() &&
        !ctx->device->fp16 && !ctx->device->coopmat2 && !ctx->device->coopmat_support &&
        src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {

        const bool live_shape   = (ne11 == 197 && ne12 * ne13 == 2);
        const bool offline_shape = (ne11 == 394 && ne12 * ne13 == 1);
        const bool supported_n  = live_shape || offline_shape;

        if (supported_n) {
            unsigned bit = 0;

            // Gate/up (M=5120 or 1280, K=1280) → prefer 32×32 when available.
            // Attention-output projection (M=1280, K=1280) also benefits.
            if (ne10 == 1280 && (ne01 == 5120 || ne01 == 1280) &&
                ctx->device->mul_mat_s[GGML_TYPE_F32]) {
                pipeline = aligned ? mmp->a_s : mmp->s;
                bit = (ne01 == 5120) ? 1u : 2u;
            }

            // Optional down-projection override (preserves original split-K).
            static const bool down_tiles = [] {
                const char *value = getenv("GEMX_VITPOSE_DOWN_TILES");
                return !value || strcmp(value, "0") != 0;
            }();
            if (down_tiles && aligned && ne10 == 5120 && ne01 == 1280 &&
                ctx->device->gemx_vitpose_down_pipeline) {
                gemx_vitpose_split_pipeline = pipeline;
                pipeline = ctx->device->gemx_vitpose_down_pipeline;
                bit = 4u;
            }

            // Experimental QKV path (M=3840, K=1280).
            // Prefer a lower-register tile when the environment requests it.
            // Default remains upstream selection for safety; enable with
            // GEMX_VITPOSE_QKV_TILES=1.
            static const bool qkv_tiles = [] {
                const char *value = getenv("GEMX_VITPOSE_QKV_TILES");
                return value && strcmp(value, "1") == 0;
            }();
            if (qkv_tiles && ne10 == 1280 && ne01 == 3840 &&
                ctx->device->gemx_qkv_pipeline) {
                pipeline = ctx->device->gemx_qkv_pipeline;
                bit |= 8u;
            }

            static const bool trace = getenv("GEMX_VITPOSE_TILES_TRACE") != nullptr;
            static thread_local unsigned traced = 0;
            if (trace && bit && !(traced & bit)) {
                GGML_LOG_INFO(
                    "gemx_vitpose_tiles: m=%llu n=%llu k=%llu batch=%llu "
                    "tile_bits=0x%x device=\"%s\"\n",
                    static_cast<unsigned long long>(ne01),
                    static_cast<unsigned long long>(ne11),
                    static_cast<unsigned long long>(ne10),
                    static_cast<unsigned long long>(ne12 * ne13),
                    bit,
                    ctx->device->properties.deviceName.data());
                traced |= bit;
            }
        }
    }
}

// ------------------------------------------------------------------
// Legacy experimental rectangular / group overrides (kept for A/B)
// ------------------------------------------------------------------
static const bool gemx_custom_tiles_enabled = [] {
    const char *value = getenv("GEMX_VITPOSE_TILES");
    return !value || strcmp(value, "0") != 0;
}();

if (gemx_custom_tiles_enabled && ctx->device->gemx_rect_pipeline && aligned &&
    ne11 == 394 && ne12 * ne13 == 1 &&
    src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 &&
    (ne10 == 1280 || ne10 == 5120)) {
    const char *group = getenv("GEMX_VITPOSE_RECT_GROUP");
    if (!group) group = "up";
    if ((strcmp(group, "down") == 0 && ne10 == 5120 && ne01 == 1280) ||
        (strcmp(group, "up")   == 0 && ne10 == 1280 && (ne01 == 5120 || ne01 == 10240)) ||
        (strcmp(group, "qkv")  == 0 && ne01 == 3840) ||
        (strcmp(group, "proj") == 0 && ne10 == 1280 && ne01 == 1280)) {
        pipeline = ctx->device->gemx_rect_pipeline;
    }
}

if (gemx_custom_tiles_enabled && ctx->device->gemx_qkv_pipeline && aligned &&
    ne10 == 1280 && ne01 == 3840 && ne11 == 394 && ne12 * ne13 == 1 &&
    src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
    // Only override when explicitly requested (see GEMX_VITPOSE_QKV_TILES above).
    if (getenv("GEMX_VITPOSE_QKV_TILES") && strcmp(getenv("GEMX_VITPOSE_QKV_TILES"), "1") == 0)
        pipeline = ctx->device->gemx_qkv_pipeline;
}

// Diagnostic: expose selected QKV pipeline statistics once.
if (getenv("GEMX_VITPOSE_TILES_TRACE") && ne10 == 1280 && ne01 == 3840 && ne11 == 394) {
    static thread_local bool qkv_traced = false;
    if (!qkv_traced) {
        GGML_LOG_INFO("gemx_qkv_pipeline: name=%s tile=%ux%u registers=%u\n",
                      pipeline->name.c_str(),
                      pipeline->wg_denoms[0], pipeline->wg_denoms[1],
                      pipeline->register_count);
        qkv_traced = true;
    }
}
