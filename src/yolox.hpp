#pragma once

#include "backend.hpp"
#include "gemx.h"
#include "model.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gemx {
std::vector<float> yolox_prepare_rgb(const gemx_rgb_frame &frame,float &ratio);
class yolox {
public:
    explicit yolox(const gemx_session_config &config);
    ~yolox();
    yolox(const yolox &)=delete;
    yolox &operator=(const yolox &)=delete;
    std::vector<gemx_detection> detect(const gemx_rgb_frame &frame,float threshold,float nms_threshold);
    const std::string &device() const{return backend_->description();}
private:
    struct graph_state;
    std::unique_ptr<backend> backend_;
    std::unique_ptr<model> model_;
    std::unique_ptr<graph_state> graph_;
    std::mutex mutex_;
};
}
