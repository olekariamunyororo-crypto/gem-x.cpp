#include "gemx.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
template<class T> void read(std::istream &in,T *value,size_t count=1){
    if(!in.read(reinterpret_cast<char *>(value),sizeof(T)*count))throw std::runtime_error("truncated packed image");
}
struct image {uint32_t width=0,height=0,stride=0;std::vector<uint8_t> rgb;};
image load(const char *path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open packed image");
    std::array<char,8> magic{};read(in,magic.data(),8);if(std::string(magic.data(),8)!="S3DIMG01")throw std::runtime_error("bad packed image");
    image result;read(in,&result.width);read(in,&result.height);read(in,&result.stride);
    std::array<float,8> ignored{};read(in,ignored.data(),ignored.size());
    result.rgb.resize(uint64_t(result.stride)*result.height);read(in,result.rgb.data(),result.rgb.size());return result;
}
}
int main(int argc,char **argv){
    try{
        if(argc!=5)throw std::runtime_error("usage: test MODEL MODULE CPU|Vulkan IMAGE");
        auto image=load(argv[4]);gemx_rgb_frame frame{image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,{}};
        gemx_session_config config{argv[1],argv[2],argv[3],"",0,2,1};char error[512]{};gemx_yolox *raw=nullptr;
        if(gemx_yolox_create(&config,&raw,error,sizeof(error))!=GEMX_OK)throw std::runtime_error(error);
        std::array<gemx_detection,100> detections{};uint32_t count=0;
        auto status=gemx_yolox_detect(raw,&frame,.1f,.5f,detections.data(),detections.size(),&count,error,sizeof(error));
        gemx_yolox_destroy(raw);if(status!=GEMX_OK)throw std::runtime_error(error);
        std::printf("detections=%u\n",count);
        for(uint32_t i=0;i<count;++i)std::printf("%u %.9g %.9g %.9g %.9g %.9g\n",i,
            detections[i].box[0],detections[i].box[1],detections[i].box[2],detections[i].box[3],detections[i].score);
        if(!count)throw std::runtime_error("YOLOX found no person");
        constexpr std::array<float,5> expected{574.20544f,-.78808665f,1664.61963f,1491.02759f,.94968653f};
        float maximum=0;
        for(size_t i=0;i<5;++i){const float actual=i<4?detections[0].box[i]:detections[0].score;maximum=std::max(maximum,std::abs(actual-expected[i]));}
        std::printf("primary max_error=%g\n",maximum);
        const float tolerance=std::strcmp(argv[3],"CPU")==0?.1f:2.f;
        if(maximum>tolerance)throw std::runtime_error("YOLOX detection differs from ONNX Runtime");
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
