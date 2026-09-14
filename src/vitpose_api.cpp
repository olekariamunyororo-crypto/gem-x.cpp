#include "vitpose.hpp"
#include "internal.hpp"
#include <algorithm>
#include <memory>

struct gemx_vitpose {std::unique_ptr<gemx::vitpose> implementation;};

gemx_status gemx_vitpose_create(const gemx_session_config *config,gemx_vitpose **output,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(output,"ViTPose output is required");*output=nullptr;
        gemx::require(config,"ViTPose configuration is required");
        auto result=std::make_unique<gemx_vitpose>();
        result->implementation=std::make_unique<gemx::vitpose>(*config);
        *output=result.release();
    });
}

void gemx_vitpose_destroy(gemx_vitpose *value){delete value;}

const char *gemx_vitpose_device(const gemx_vitpose *value){
    return value && value->implementation?value->implementation->device().c_str():nullptr;
}

gemx_status gemx_vitpose_prepare_rgb(const gemx_rgb_frame *frame,float *output,
    uint64_t count,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(frame && output && count==uint64_t(3)*256*192,
                      "ViTPose frame and 3x256x192 output are required");
        auto value=gemx::vitpose_prepare_rgb(*frame);
        std::copy(value.begin(),value.end(),output);
    });
}

gemx_status gemx_vitpose_infer_normalized(gemx_vitpose *value,const float *images,
    uint32_t batch,float *heatmaps,uint64_t count,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(value && value->implementation && images && heatmaps,
                      "ViTPose model, images and heatmaps are required");
        gemx::require(count==uint64_t(batch)*77*64*48,
                      "ViTPose heatmap count does not match batch");
        value->implementation->infer_normalized(images,batch,heatmaps);
    });
}

gemx_status gemx_vitpose_infer_rgb(gemx_vitpose *value,const gemx_rgb_frame *frames,
    uint32_t frame_count,float *keypoints,uint64_t count,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(value && value->implementation && frames && keypoints,
                      "ViTPose model, frames and keypoint output are required");
        gemx::require(count==uint64_t(frame_count)*77*3,
                      "ViTPose keypoint count does not match frames");
        value->implementation->infer_rgb(frames,frame_count,keypoints);
    });
}
