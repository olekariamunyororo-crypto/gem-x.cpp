#include "gemx.h"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main(){
    const float r[6]={1,0,0,0,1,0};float m[9];char error[128];
    if(gemx_rotation_6d_to_matrix(r,m,error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
    for(uint32_t i=0;i<9;++i)if(std::abs(m[i]-(i%4==0?1.f:0.f))>1e-7f)throw std::runtime_error("identity rotation differs");
    const float degenerate[6]={1,0,0,2,0,0};
    if(gemx_rotation_6d_to_matrix(degenerate,m,error,sizeof error)!=GEMX_INVALID_ARGUMENT)throw std::runtime_error("degenerate rotation accepted");
    std::array<float,GEMX_MOTION_DIM> input{},mean{},stddev{},output{};
    for(uint32_t i=0;i<GEMX_MOTION_DIM;++i){input[i]=float(i%7);mean[i]=float(i%3);stddev[i]=2;}
    if(gemx_denormalize_motion(input.data(),1,mean.data(),stddev.data(),output.data(),error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
    for(uint32_t i=0;i<GEMX_MOTION_DIM;++i)if(output[i]!=input[i]*2+mean[i])throw std::runtime_error("motion denormalization differs");
    std::cout<<"GEM-X rotation and motion decoding contract passed\n";
}

