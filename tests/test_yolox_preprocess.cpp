#include "gemx.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main(){
    try{
        // FNV-1a hashes of the focused uint8 pixels generated independently by
        // OpenCV 4.11.0.86 INTER_LINEAR, BGR conversion and 114 letterboxing.
        // Input RGB channels are (17x+3y+11, 5x+13y+29, 7x+19y+47) modulo 256.
        struct sample {uint32_t width,height;uint64_t hash;};
        constexpr std::array<sample,5> cases{{{480,270,0x6f252f4332452f6cULL},
            {317,191,0x9921c686cd36c697ULL},{641,359,0x8f3f48e165339eefULL},
            {13,7,0xe2de9c7b42e47622ULL},{640,640,0xd4fbd9d862d49125ULL}}};
        for(const auto &s:cases){
            std::vector<uint8_t> rgb(uint64_t(s.width)*s.height*3);
            for(uint32_t y=0;y<s.height;++y)for(uint32_t x=0;x<s.width;++x){
                const auto at=(uint64_t(y)*s.width+x)*3;
                rgb[at]=(17*x+3*y+11)%256;rgb[at+1]=(5*x+13*y+29)%256;rgb[at+2]=(7*x+19*y+47)%256;
            }
            gemx_rgb_frame frame{rgb.data(),rgb.size(),s.width,s.height,uint64_t(s.width)*3,{}};
            std::vector<float> output(12*320*320);float ratio=0;char error[512]{};
            if(gemx_yolox_prepare_rgb(&frame,output.data(),output.size(),&ratio,error,sizeof(error))!=GEMX_OK)
                throw std::runtime_error(error);
            uint64_t hash=14695981039346656037ULL;
            for(float value:output){
                if(!std::isfinite(value)||value<0||value>255||value!=std::floor(value))
                    throw std::runtime_error("invalid resized pixel");
                hash=(hash^uint8_t(value))*1099511628211ULL;
            }
            if(hash!=s.hash)throw std::runtime_error("YOLOX resize differs from OpenCV reference");
            if(std::abs(ratio-float(std::min(640./s.width,640./s.height)))>1e-7f)
                throw std::runtime_error("incorrect resize ratio");
        }
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
