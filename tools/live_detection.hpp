#pragma once
#include <cstdint>

// Count processed frames, not camera frames. Misses retry immediately.
class live_detection_schedule {
    uint32_t interval_, remaining_ = 0;
public:
    explicit live_detection_schedule(uint32_t interval): interval_(interval) {}
    bool due() {
        if (!remaining_) return true;
        --remaining_;
        return false;
    }
    void detected(bool found) { remaining_ = found ? interval_ - 1 : 0; }
    void reset() { remaining_ = 0; }
};
