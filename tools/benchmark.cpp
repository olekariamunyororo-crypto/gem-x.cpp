#include "gemx.h"
#include <array>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using file_ptr=std::unique_ptr<FILE,int(*)(FILE *)>;
void read(FILE *file,void *data,size_t size){
    if(std::fread(data,1,size,file)!=size)throw std::runtime_error("truncated reference fixture");
}
uint32_t number(const char *text){
    uint32_t value=0;const char *end=text+std::strlen(text);
    auto result=std::from_chars(text,end,value);
    if(result.ec!=std::errc{} || result.ptr!=end)throw std::invalid_argument("invalid integer argument");
    return value;
}
std::vector<float> floats(FILE *file,size_t count){std::vector<float> value(count);read(file,value.data(),count*4);return value;}
}

int main(int argc,char **argv){
    if(argc!=9 && argc!=10){
        std::fprintf(stderr,"usage: %s MODEL BACKEND_MODULE CPU|Vulkan DEVICE THREADS FIXTURE FRAMES ITERATIONS [--live]\n",argv[0]);
        return 2;
    }
    try{
        const uint32_t device=number(argv[4]),threads=number(argv[5]);
        const uint32_t frames=number(argv[7]),iterations=number(argv[8]);
        const bool live=argc==10 && std::strcmp(argv[9],"--live")==0;
        if(argc==10 && !live)throw std::invalid_argument("unknown benchmark mode");
        if(frames<1 || frames>120 || iterations<1 || iterations>10000)
            throw std::invalid_argument("frames must be 1..120 and iterations 1..10000");
        file_ptr file(std::fopen(argv[6],"rb"),std::fclose);
        if(!file)throw std::runtime_error("cannot open reference fixture");
        char magic[8];uint32_t header[7];read(file.get(),magic,8);read(file.get(),header,sizeof(header));
        if(std::memcmp(magic,"GEMXREF1",8) || header[0]!=1 || header[1]!=120 ||
           header[3]!=77 || header[4]!=1024)throw std::runtime_error("unsupported reference fixture");
        std::vector<uint32_t> lengths(header[2]);read(file.get(),lengths.data(),lengths.size()*4);
        auto keypoints=floats(file.get(),uint64_t(header[1])*77*3);
        auto boxes=floats(file.get(),uint64_t(header[1])*3);
        auto intrinsics=floats(file.get(),uint64_t(header[1])*9);
        auto features=floats(file.get(),uint64_t(header[1])*1024);
        auto angular=floats(file.get(),uint64_t(header[1])*6);
        gemx_sequence_view input{frames,keypoints.data(),boxes.data(),intrinsics.data(),features.data(),angular.data()};
        gemx_session_config config{argv[1],argv[2],argv[3],nullptr,device,threads,2};
        gemx_session *raw=nullptr;char error[512]{};
        auto status=gemx_session_create(&config,&raw,error,sizeof(error));
        if(status!=GEMX_OK)throw std::runtime_error(error);
        std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)> session(raw,gemx_session_destroy);
        std::vector<float> motion(uint64_t(frames)*585),camera(uint64_t(frames)*3);
        std::unique_ptr<gemx_live,decltype(&gemx_live_destroy)> stream(nullptr,gemx_live_destroy);
        std::array<float,585> newest_motion{};std::array<float,3> newest_camera{};
        if(live){
            gemx_live *raw_live=nullptr;status=gemx_live_create(session.get(),frames,&raw_live,error,sizeof(error));
            if(status!=GEMX_OK)throw std::runtime_error(error);stream.reset(raw_live);
            gemx_sequence_view one{1,keypoints.data(),boxes.data(),intrinsics.data(),features.data(),angular.data()};
            status=gemx_live_push(stream.get(),&one,newest_motion.data(),newest_camera.data(),error,sizeof(error));
        }else status=gemx_infer(session.get(),&input,motion.data(),motion.size(),camera.data(),camera.size(),error,sizeof(error));
        if(status!=GEMX_OK)throw std::runtime_error(error);
        gemx_profile before{},after{};
        gemx_session_get_profile(session.get(),&before,error,sizeof(error));
        const auto started=std::chrono::steady_clock::now();
        for(uint32_t i=0;i<iterations;++i){
            if(live){
                const uint32_t frame=i%frames;
                gemx_sequence_view one{1,keypoints.data()+uint64_t(frame)*77*3,boxes.data()+uint64_t(frame)*3,
                    intrinsics.data()+uint64_t(frame)*9,features.data()+uint64_t(frame)*1024,
                    angular.data()+uint64_t(frame)*6};
                status=gemx_live_push(stream.get(),&one,newest_motion.data(),newest_camera.data(),error,sizeof(error));
            }else status=gemx_infer(session.get(),&input,motion.data(),motion.size(),camera.data(),camera.size(),error,sizeof(error));
            if(status!=GEMX_OK)throw std::runtime_error(error);
        }
        const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()/iterations;
        gemx_session_get_profile(session.get(),&after,error,sizeof(error));
        auto milliseconds=[&](uint64_t gemx_profile::*field){return double(after.*field-before.*field)/1e6/iterations;};
        std::printf("mode=%s device=%s frames=%u threads=%u iterations=%u wall_ms=%.3f throughput_fps=%.2f "
                    "pre_ms=%.3f upload_ms=%.3f infer_ms=%.3f download_ms=%.3f cache_hits=%llu\n",
                    live?"live":"offline",gemx_session_device(session.get()),frames,threads,iterations,wall_ms,
                    (live?1:frames)*1000./wall_ms,
                    milliseconds(&gemx_profile::preprocessing_ns),milliseconds(&gemx_profile::upload_ns),
                    milliseconds(&gemx_profile::inference_ns),milliseconds(&gemx_profile::download_ns),
                    static_cast<unsigned long long>(after.graph_cache_hits-before.graph_cache_hits));
        return 0;
    }catch(const std::exception &exception){std::fprintf(stderr,"%s\n",exception.what());return 1;}
}
