#pragma once

#include "backend.hpp"
#include "gemx.h"
#include "model.hpp"
#include "skeleton.hpp"
#include "soma_identity.hpp"
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gemx {
class session {
public:
    explicit session(const gemx_session_config &config);
    ~session();
    void infer(const gemx_sequence_view &input,float *motion,float *camera);
    void infer_live_frame(uint32_t context,uint32_t slot,uint32_t count,uint32_t next,
                          const gemx_sequence_view &frame,float *motion,float *camera);
    void decode(const gemx_sequence_view &input,const float *motion,const float *camera,
                const gemx_motion_view &output);
    void decode_velocity(const float *motion,float velocity[3]) const;
    skeleton_data skeleton(const gemx_motion_view &motion) const;
    void export_skeleton(const gemx_skeleton_view &skeleton,float fps,const std::string &path) const;
    const std::string &device() const{return backend_->description();}
    gemx_profile profile() const;
private:
    struct graph_state;
    graph_state &graph(uint32_t frames,bool live=false);
    void infer_window(const gemx_sequence_view &input,float *motion,float *camera);
    std::unique_ptr<backend> backend_;
    std::unique_ptr<model> model_;
    uint32_t cache_capacity_=4;
    std::unordered_map<uint32_t,std::unique_ptr<graph_state>> graphs_;
    std::list<uint32_t> lru_;
    mutable std::mutex mutex_;
    gemx_profile profile_{};
    std::vector<float> motion_mean_,motion_std_;
    std::vector<float> soma_local_,soma_world_;
    std::vector<int32_t> soma_parents_;
    std::vector<std::string> soma_names_;
    soma_identity_constants soma_identity_;
};
}
