#include "internal.hpp"
#include <algorithm>
#include <limits>

gemx_status gemx_preprocess_sequence(const gemx_sequence_view *in,
    float *normalized,uint64_t normalized_count,float *cliff,uint64_t cliff_count,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(in && normalized && cliff,"input and output buffers required");
        gemx::require(in->frames>=1 && in->frames<=GEMX_MAX_FRAMES,"frames must be in 1..4096");
        gemx::require(in->keypoints && in->boxes && in->intrinsics && in->body_features && in->camera_angular_velocity,"all sequence arrays required");
        const uint64_t kp_count=uint64_t(in->frames)*GEMX_SOMA_JOINTS*3;
        gemx::require(normalized_count==kp_count && cliff_count==uint64_t(in->frames)*3,"output counts do not match sequence");
        for(uint32_t f=0;f<in->frames;++f){
            const float *box=in->boxes+uint64_t(f)*3,*K=in->intrinsics+uint64_t(f)*9;
            gemx::require(gemx::finite(box[0]) && gemx::finite(box[1]) && gemx::finite(box[2]) && box[2]>0,"finite positive boxes required");
            for(uint32_t i=0;i<9;++i)gemx::require(gemx::finite(K[i]),"finite intrinsics required");
            gemx::require(K[0]>0 && K[4]>0,"positive focal lengths required");
            for(uint32_t i=0;i<GEMX_BODY_FEATURE_DIM;++i)gemx::require(gemx::finite(in->body_features[uint64_t(f)*GEMX_BODY_FEATURE_DIM+i]),"finite Body features required");
            for(uint32_t i=0;i<6;++i)gemx::require(gemx::finite(in->camera_angular_velocity[uint64_t(f)*6+i]),"finite camera angular velocity required");
            const float scale=std::max(box[2],.01f),half=box[2]*.5f;
            for(uint32_t j=0;j<GEMX_SOMA_JOINTS;++j){
                const float *src=in->keypoints+(uint64_t(f)*GEMX_SOMA_JOINTS+j)*3;
                float *dst=normalized+(uint64_t(f)*GEMX_SOMA_JOINTS+j)*3;
                gemx::require(gemx::finite(src[0]) && gemx::finite(src[1]) && gemx::finite(src[2]),"finite keypoints required");
                dst[0]=2*(src[0]-box[0])/scale;dst[1]=2*(src[1]-box[1])/scale;
                const bool outside=src[0]<box[0]-half || src[0]>box[0]+half || src[1]<box[1]-half || src[1]>box[1]+half;
                dst[2]=outside?0.f:src[2];
            }
            cliff[uint64_t(f)*3]=(box[0]-K[2])/K[0];
            cliff[uint64_t(f)*3+1]=(box[1]-K[5])/K[0];
            cliff[uint64_t(f)*3+2]=box[2]/K[0];
            gemx::require(gemx::finite(cliff[uint64_t(f)*3]) && gemx::finite(cliff[uint64_t(f)*3+1]) && gemx::finite(cliff[uint64_t(f)*3+2]),"camera condition overflow");
        }
    });
}

