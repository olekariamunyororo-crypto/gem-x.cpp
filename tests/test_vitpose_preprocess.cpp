#include "gemx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {
using file_ptr=std::unique_ptr<FILE,int(*)(FILE *)>;
void read(FILE *file,void *data,size_t bytes){if(std::fread(data,1,bytes,file)!=bytes)throw std::runtime_error("truncated fixture");}
}

int main(int argc,char **argv){
    if(argc!=2){std::fprintf(stderr,"usage: %s FIXTURE\n",argv[0]);return 2;}
    try{
        file_ptr file(std::fopen(argv[1],"rb"),std::fclose);if(!file)throw std::runtime_error("cannot open fixture");
        char magic[8];read(file.get(),magic,8);if(std::memcmp(magic,"VITPREP1",8))throw std::runtime_error("invalid fixture");
        uint32_t width,height;float box[3];read(file.get(),&width,4);read(file.get(),&height,4);read(file.get(),box,sizeof box);
        std::vector<float> expected(uint64_t(3)*256*192);read(file.get(),expected.data(),expected.size()*4);
        std::vector<uint8_t> rgb(uint64_t(width)*height*3);
        for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){
            auto *pixel=rgb.data()+(uint64_t(y)*width+x)*3;
            pixel[0]=uint8_t((x*17+y*3+11)%256);pixel[1]=uint8_t((x*5+y*13+29)%256);
            pixel[2]=uint8_t((x*7+y*19+47)%256);}
        gemx_rgb_frame input{rgb.data(),rgb.size(),width,height,uint64_t(width)*3,{box[0],box[1],box[2]}};
        std::vector<float> actual(expected.size());char error[256]{};
        if(gemx_vitpose_prepare_rgb(&input,actual.data(),actual.size(),error,sizeof error)!=GEMX_OK)
            throw std::runtime_error(error);
        float maximum=0;double mean=0;uint64_t changed=0;
        for(size_t i=0;i<actual.size();++i){const float difference=std::abs(actual[i]-expected[i]);
            maximum=std::max(maximum,difference);mean+=difference;changed+=difference!=0;}
        mean/=actual.size();std::printf("ViTPose/OpenCV preprocess max=%g mean=%g changed=%llu\n",
            maximum,mean,static_cast<unsigned long long>(changed));
        if(maximum>1e-6f)throw std::runtime_error("ViTPose preprocessing differs from OpenCV reference");
        return 0;
    }catch(const std::exception &exception){std::fprintf(stderr,"%s\n",exception.what());return 1;}
}
