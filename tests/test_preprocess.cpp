#include "gemx.h"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(){
    std::array<float,GEMX_SOMA_JOINTS*3> kp{};
    for(uint32_t j=0;j<GEMX_SOMA_JOINTS;++j){kp[j*3]=100;kp[j*3+1]=50;kp[j*3+2]=.75f;}
    kp[3]=151;kp[4]=50;kp[5]=.9f;
    const float box[3]={100,50,100},K[9]={200,0,80,0,210,60,0,0,1};
    std::array<float,GEMX_BODY_FEATURE_DIM> feature{};const float angular[6]={1,0,0,0,1,0};
    gemx_sequence_view view{1,kp.data(),box,K,feature.data(),angular};
    std::array<float,GEMX_SOMA_JOINTS*3> out{};float cliff[3]={};char error[128];
    if(gemx_preprocess_sequence(&view,out.data(),out.size(),cliff,3,error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
    auto near=[](float a,float b){return std::abs(a-b)<1e-7f;};
    if(!near(out[0],0)||!near(out[1],0)||!near(out[2],.75f)||!near(out[3],1.02f)||!near(out[5],0)||
       !near(cliff[0],.1f)||!near(cliff[1],-.05f)||!near(cliff[2],.5f))throw std::runtime_error("preprocessing values differ");
    auto bad=view;bad.frames=0;if(gemx_preprocess_sequence(&bad,out.data(),out.size(),cliff,3,error,sizeof error)!=GEMX_INVALID_ARGUMENT)throw std::runtime_error("zero frames accepted");
    const float zero_angular[6]={};bad=view;bad.camera_angular_velocity=zero_angular;
    if(gemx_preprocess_sequence(&bad,out.data(),out.size(),cliff,3,error,sizeof error)!=GEMX_INVALID_ARGUMENT)throw std::runtime_error("degenerate camera rotation accepted");
    auto bad_kp=kp;bad_kp[0]=std::numeric_limits<float>::quiet_NaN();bad=view;bad.keypoints=bad_kp.data();
    if(gemx_preprocess_sequence(&bad,out.data(),out.size(),cliff,3,error,sizeof error)!=GEMX_INVALID_ARGUMENT)throw std::runtime_error("nonfinite keypoint accepted");
    std::cout<<"GEM-X keypoint and camera preprocessing contract passed\n";
}
