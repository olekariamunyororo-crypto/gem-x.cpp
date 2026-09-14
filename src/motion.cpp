#include "internal.hpp"
#include <algorithm>
#include <cmath>
#include <span>

namespace {
void normalize3(float v[3]){
    const double n=std::sqrt(double(v[0])*v[0]+double(v[1])*v[1]+double(v[2])*v[2]);
    gemx::require(std::isfinite(n) && n>1e-12,"degenerate 6D rotation");
    for(float &x:std::span<float,3>(v,3))x=float(x/n);
}
}
gemx_status gemx_rotation_6d_to_matrix(const float r[6],float m[9],char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(r && m,"rotation and matrix required");
        for(uint32_t i=0;i<6;++i)gemx::require(gemx::finite(r[i]),"finite rotation required");
        float a[3]={r[0],r[1],r[2]};normalize3(a);
        const float dot=a[0]*r[3]+a[1]*r[4]+a[2]*r[5];
        float b[3]={r[3]-dot*a[0],r[4]-dot*a[1],r[5]-dot*a[2]};normalize3(b);
        const float c[3]={a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};
        std::copy_n(a,3,m);std::copy_n(b,3,m+3);std::copy_n(c,3,m+6);
    });
}
gemx_status gemx_denormalize_motion(const float *input,uint32_t frames,const float *mean,const float *stddev,float *output,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(input && mean && stddev && output,"motion buffers and statistics required");
        gemx::require(frames>=1 && frames<=GEMX_MAX_FRAMES,"frames must be in 1..4096");
        for(uint32_t i=0;i<GEMX_MOTION_DIM;++i)gemx::require(gemx::finite(mean[i]) && gemx::finite(stddev[i]) && stddev[i]>0,"finite positive motion statistics required");
        const uint64_t count=uint64_t(frames)*GEMX_MOTION_DIM;
        for(uint64_t i=0;i<count;++i){
            gemx::require(gemx::finite(input[i]),"finite normalized motion required");
            output[i]=input[i]*stddev[i%GEMX_MOTION_DIM]+mean[i%GEMX_MOTION_DIM];
            gemx::require(gemx::finite(output[i]),"motion denormalization overflow");
        }
    });
}
