#pragma once

#include "backend.hpp"
#include "gemx.h"
#include "model.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace gemx {
std::vector<float> vitpose_prepare_rgb(const gemx_rgb_frame &frame);
class vitpose {
public:
    explicit vitpose(const gemx_session_config &config);
    ~vitpose();
    vitpose(const vitpose &)=delete;
    vitpose &operator=(const vitpose &)=delete;
    void infer_normalized(const float *images,uint32_t batch,float *heatmaps);
    void infer_rgb(const gemx_rgb_frame *frames,uint32_t count,float *keypoints);
    const std::string &device() const{return backend_->description();}
private:
    struct graph_state;
    graph_state &graph(uint32_t batch);
    std::unique_ptr<backend> backend_;
    std::unique_ptr<model> model_;
    std::unordered_map<uint32_t,std::unique_ptr<graph_state>> graphs_;
    std::mutex mutex_;
};
}
