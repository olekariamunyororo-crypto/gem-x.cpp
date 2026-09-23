// Optional adaptive fence wait for ggml_vk_wait_for_fence.
// Included immediately before the function definition by the CMake adapter.
//
// When GEMX_FENCE_POLL_SLEEP=1 the busy-pause spin is replaced by a short
// adaptive sleep after a busy window. Default behaviour (variable unset or 0)
// remains the original high-frequency YIELD() loop.

#include <chrono>
#include <thread>
#include <cstdlib>
#include <cstring>

namespace gemx_fence {
inline bool sleep_enabled() {
    static const bool enabled = [] {
        const char *v = getenv("GEMX_FENCE_POLL_SLEEP");
        return v && strcmp(v, "1") == 0;
    }();
    return enabled;
}

inline void adaptive_pause(int iter) {
    constexpr int kBusyIters = 64;
    constexpr auto kSleep = std::chrono::microseconds(50);
    if (sleep_enabled() && iter >= kBusyIters) {
        std::this_thread::sleep_for(kSleep);
    }
}
} // namespace gemx_fence
