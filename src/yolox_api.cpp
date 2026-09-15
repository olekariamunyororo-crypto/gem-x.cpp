#include "yolox.hpp"
#include "internal.hpp"
#include <algorithm>
#include <memory>

struct gemx_yolox {std::unique_ptr<gemx::yolox> implementation;};

gemx_status gemx_yolox_create(const gemx_session_config *config,gemx_yolox **output,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(output,"YOLOX output is required");*output=nullptr;
        gemx::require(config,"YOLOX configuration is required");
        auto result=std::make_unique<gemx_yolox>();
        result->implementation=std::make_unique<gemx::yolox>(*config);
        *output=result.release();
    });
}

void gemx_yolox_destroy(gemx_yolox *value){delete value;}

const char *gemx_yolox_device(const gemx_yolox *value){
    return value&&value->implementation?value->implementation->device().c_str():nullptr;
}

gemx_status gemx_yolox_prepare_rgb(const gemx_rgb_frame *frame,float *output,
    uint64_t count,float *ratio,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(frame&&output&&ratio&&count==uint64_t(12)*320*320,
                      "YOLOX frame, ratio and 12x320x320 output are required");
        auto value=gemx::yolox_prepare_rgb(*frame,*ratio);
        std::copy(value.begin(),value.end(),output);
    });
}

gemx_status gemx_yolox_detect(gemx_yolox *value,const gemx_rgb_frame *frame,
    float score_threshold,float nms_threshold,gemx_detection *detections,
    uint32_t capacity,uint32_t *count,char *error,uint64_t error_capacity){
    return gemx::boundary(error,error_capacity,[&]{
        gemx::require(value&&value->implementation&&frame&&count,
                      "YOLOX model, frame and count are required");
        gemx::require(score_threshold>=0.f&&score_threshold<=1.f&&nms_threshold>=0.f&&nms_threshold<=1.f,
                      "YOLOX thresholds must be in 0..1");
        auto result=value->implementation->detect(*frame,score_threshold,nms_threshold);
        gemx::require(result.size()<=capacity&&(!result.size()||detections),
                      "YOLOX detection capacity is too small");
        std::copy(result.begin(),result.end(),detections);*count=static_cast<uint32_t>(result.size());
    });
}
