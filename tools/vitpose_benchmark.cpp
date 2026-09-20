#include "gemx.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <ctime>

static double seconds(){
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc,char **argv){
    if(argc!=7){std::fprintf(stderr,"usage: %s MODEL MODULE CPU|Vulkan THREADS BATCH ITERATIONS\n",argv[0]);return 2;}
    try{
        const uint32_t threads=std::stoul(argv[4]),batch=std::stoul(argv[5]),iterations=std::stoul(argv[6]);
        if(batch<1||batch>8||!iterations)throw std::invalid_argument("invalid batch or iteration count");
        gemx_session_config config{argv[1],argv[2],argv[3],nullptr,0,threads,2};char error[512]{};gemx_vitpose *raw=nullptr;
        const auto load_start=std::chrono::steady_clock::now();
        if(gemx_vitpose_create(&config,&raw,error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
        std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)> model(raw,gemx_vitpose_destroy);
        const double load_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-load_start).count();
        std::vector<float> images(uint64_t(batch)*3*256*192),heatmaps(uint64_t(batch)*77*64*48);
        for(size_t i=0;i<images.size();++i)images[i]=float(int(i%251)-125)/64.f;
        if(const char *path=std::getenv("GEMX_BENCHMARK_INPUT")){
            std::unique_ptr<FILE,int(*)(FILE *)> input(std::fopen(path,"rb"),std::fclose);
            if(!input || std::fread(images.data(),sizeof(float),images.size(),input.get())!=images.size() ||
               std::fgetc(input.get())!=EOF)throw std::runtime_error("invalid normalized benchmark input");
        }
        const char *warmup_env=std::getenv("GEMX_BENCHMARK_WARMUP");
        const uint32_t warmups=warmup_env?std::stoul(warmup_env):1;
        if(warmups>1000)throw std::invalid_argument("too many warmups");
        for(uint32_t i=0;i<warmups;++i)
            if(gemx_vitpose_infer_normalized(model.get(),images.data(),batch,heatmaps.data(),heatmaps.size(),error,sizeof error)!=GEMX_OK)
                throw std::runtime_error(error);
        std::unique_ptr<FILE,int(*)(FILE *)> profile(nullptr,std::fclose);
        if(const char *path=std::getenv("GEMX_BENCHMARK_PROFILE")){
            profile.reset(std::fopen(path,"w"));
            if(!profile)throw std::runtime_error("cannot open benchmark profile");
            std::fprintf(profile.get(),"iteration,start_seconds,end_seconds,wall_ms,cpu_ms\n");
        }
        const auto start=std::chrono::steady_clock::now();
        for(uint32_t i=0;i<iterations;++i){
            const double wall=profile?seconds():0;
            const auto cpu=profile?std::clock():0;
            if(gemx_vitpose_infer_normalized(model.get(),images.data(),batch,
                heatmaps.data(),heatmaps.size(),error,sizeof error)!=GEMX_OK)throw std::runtime_error(error);
            if(profile){
                const double end=seconds();const auto cpu_end=std::clock();
                std::fprintf(profile.get(),"%u,%.9f,%.9f,%.6f,%.6f\n",i,wall,end,(end-wall)*1000,
                    double(cpu_end-cpu)*1000/CLOCKS_PER_SEC);
            }
        }
        const double milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()/iterations;
        std::printf("device=%s batch=%u threads=%u load_ms=%.2f infer_ms=%.3f images_per_second=%.2f\n",
            gemx_vitpose_device(model.get()),batch,threads,load_ms,milliseconds,batch*1000./milliseconds);
        if(const char *path=std::getenv("GEMX_BENCHMARK_OUTPUT")){
            std::unique_ptr<FILE,int(*)(FILE *)> output(std::fopen(path,"wb"),std::fclose);
            if(!output || std::fwrite(heatmaps.data(),sizeof(float),heatmaps.size(),output.get())!=heatmaps.size())
                throw std::runtime_error("cannot write benchmark heatmaps");
        }
        return 0;
    }catch(const std::exception &exception){std::fprintf(stderr,"%s\n",exception.what());return 1;}
}
