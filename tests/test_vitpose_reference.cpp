#include "gemx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using file_ptr=std::unique_ptr<FILE,int(*)(FILE *)>;
void read(FILE *file,void *data,size_t bytes){if(std::fread(data,1,bytes,file)!=bytes)throw std::runtime_error("truncated fixture");}
std::vector<float> floats(FILE *file,size_t count){std::vector<float> result(count);read(file,result.data(),count*4);return result;}
struct stats {float maximum=0;double mean=0;};
stats compare(const float *actual,const float *expected,size_t count){stats result;
    for(size_t i=0;i<count;++i){if(!std::isfinite(actual[i]))throw std::runtime_error("non-finite native value");
        float difference=std::abs(actual[i]-expected[i]);result.maximum=std::max(result.maximum,difference);result.mean+=difference;}
    result.mean/=count;return result;}
}

int main(int argc,char **argv){
    if(argc!=6){std::fprintf(stderr,"usage: %s MODEL BACKEND PREPROCESS HEATMAP KEYPOINT\n",argv[0]);return 2;}
    try{
        file_ptr prep(std::fopen(argv[3],"rb"),std::fclose);char magic[8];read(prep.get(),magic,8);
        uint32_t width,height;float box[3];read(prep.get(),&width,4);read(prep.get(),&height,4);read(prep.get(),box,sizeof box);
        if(std::memcmp(magic,"VITPREP1",8))throw std::runtime_error("invalid preprocess fixture");
        auto image=floats(prep.get(),uint64_t(3)*256*192);
        std::vector<float> batch(image.size()*2);std::copy(image.begin(),image.end(),batch.begin());
        for(int c=0;c<3;++c)for(int y=0;y<256;++y)for(int x=0;x<192;++x)
            batch[image.size()+(c*256+y)*192+x]=image[(c*256+y)*192+(191-x)];
        file_ptr heat_file(std::fopen(argv[4],"rb"),std::fclose);read(heat_file.get(),magic,8);uint32_t heat_batch;
        read(heat_file.get(),&heat_batch,4);if(std::memcmp(magic,"VITHEAT1",8)||heat_batch!=2)throw std::runtime_error("invalid heatmap fixture");
        auto expected_heat=floats(heat_file.get(),uint64_t(2)*77*64*48);
        const char *name=std::getenv("GEMX_TEST_BACKEND");if(!name||!*name)name="CPU";
        gemx_session_config config{argv[1],argv[2],name,nullptr,0,2,2};char error[512]{};gemx_vitpose *raw=nullptr;
        if(gemx_vitpose_create(&config,&raw,error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
        std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)> model(raw,gemx_vitpose_destroy);
        std::vector<float> actual_heat(expected_heat.size());
        if(gemx_vitpose_infer_normalized(model.get(),batch.data(),2,actual_heat.data(),actual_heat.size(),error,sizeof error)!=GEMX_OK)
            throw std::runtime_error(error);
        auto heat_error=compare(actual_heat.data(),expected_heat.data(),actual_heat.size());
        std::printf("ViTPose heatmaps max=%g mean=%g\n",heat_error.maximum,heat_error.mean);
        const bool vulkan=std::strcmp(name,"Vulkan")==0;
        if(heat_error.maximum>(vulkan?4e-3f:3e-4f)||heat_error.mean>(vulkan?4e-4:3e-5))
            throw std::runtime_error("ViTPose heatmap divergence exceeds tolerance");
        std::vector<uint8_t> rgb(uint64_t(width)*height*3);
        for(uint32_t y=0;y<height;++y)for(uint32_t x=0;x<width;++x){auto *p=rgb.data()+(uint64_t(y)*width+x)*3;
            p[0]=uint8_t((x*17+y*3+11)%256);p[1]=uint8_t((x*5+y*13+29)%256);p[2]=uint8_t((x*7+y*19+47)%256);}
        gemx_rgb_frame frame{rgb.data(),rgb.size(),width,height,uint64_t(width)*3,{box[0],box[1],box[2]}};
        std::vector<float> actual_keypoints(77*3);if(gemx_vitpose_infer_rgb(model.get(),&frame,1,
            actual_keypoints.data(),actual_keypoints.size(),error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
        file_ptr key_file(std::fopen(argv[5],"rb"),std::fclose);read(key_file.get(),magic,8);
        if(std::memcmp(magic,"VITKEYP1",8))throw std::runtime_error("invalid keypoint fixture");
        auto expected_keypoints=floats(key_file.get(),77*3);auto key_error=compare(actual_keypoints.data(),expected_keypoints.data(),77*3);
        std::printf("ViTPose keypoints max=%g mean=%g\n",key_error.maximum,key_error.mean);
        size_t worst=0;for(size_t i=1;i<actual_keypoints.size();++i)
            if(std::abs(actual_keypoints[i]-expected_keypoints[i])>
               std::abs(actual_keypoints[worst]-expected_keypoints[worst]))worst=i;
        std::printf("worst keypoint=%zu component=%zu actual=%g expected=%g confidence=%g\n",
            worst/3,worst%3,actual_keypoints[worst],expected_keypoints[worst],
            actual_keypoints[(worst/3)*3+2]);
        uint32_t displaced=0;float worst_distance=0;
        for(size_t joint=0;joint<77;++joint){float dx=actual_keypoints[joint*3]-expected_keypoints[joint*3];
            float dy=actual_keypoints[joint*3+1]-expected_keypoints[joint*3+1];float distance=std::hypot(dx,dy);
            displaced+=distance>1.f;worst_distance=std::max(worst_distance,distance);}
        std::printf("keypoints displaced_over_1px=%u/77 worst_distance=%g px\n",displaced,worst_distance);
        if(key_error.maximum>(vulkan?7.f:.1f)||key_error.mean>(vulkan?.08:.005)||
           (vulkan && displaced>4))
            throw std::runtime_error("ViTPose keypoint divergence exceeds tolerance");
        return 0;
    }catch(const std::exception &exception){std::fprintf(stderr,"%s\n",exception.what());return 1;}
}
