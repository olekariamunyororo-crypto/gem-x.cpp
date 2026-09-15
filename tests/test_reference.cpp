#include "gemx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using file_ptr=std::unique_ptr<FILE,int(*)(FILE *)>;
void read(FILE *file,void *data,size_t size){
    if(std::fread(data,1,size,file)!=size)throw std::runtime_error("truncated reference fixture");
}
std::vector<float> read_floats(FILE *file,size_t count){
    std::vector<float> result(count);read(file,result.data(),count*sizeof(float));return result;
}
struct error_stats {double mean=0;float maximum=0;};
error_stats compare(const float *actual,const float *expected,size_t count){
    error_stats result;
    for(size_t i=0;i<count;++i){
        if(!std::isfinite(actual[i]))throw std::runtime_error("native output is non-finite");
        float difference=std::abs(actual[i]-expected[i]);
        result.maximum=std::max(result.maximum,difference);result.mean+=difference;
    }
    result.mean/=count;return result;
}
void require_finite(const std::vector<float> &values,const char *label){
    if(!std::all_of(values.begin(),values.end(),[](float value){return std::isfinite(value);}))
        throw std::runtime_error(std::string(label)+" contains non-finite values");
}
}

int main(int argc,char **argv){
    if(argc<4 || argc>6){
        std::fprintf(stderr,"usage: %s MODEL BACKEND INFERENCE_FIXTURE [POSTPROCESS_FIXTURE [SKELETON_FIXTURE]]\n",argv[0]);
        return 2;
    }
    try{
        file_ptr file(std::fopen(argv[3],"rb"),std::fclose);
        if(!file)throw std::runtime_error("cannot open reference fixture");
        char magic[8];read(file.get(),magic,8);
        if(std::memcmp(magic,"GEMXREF1",8))throw std::runtime_error("bad reference fixture magic");
        uint32_t header[7];read(file.get(),header,sizeof(header));
        if(header[0]!=1 || header[1]!=120 || header[2]!=5 || header[3]!=77 ||
           header[4]!=1024 || header[5]!=585 || header[6]!=3)
            throw std::runtime_error("unsupported reference fixture dimensions");
        std::vector<uint32_t> lengths(header[2]);read(file.get(),lengths.data(),lengths.size()*4);
        const uint32_t maximum=header[1];
        auto keypoints=read_floats(file.get(),uint64_t(maximum)*77*3);
        auto boxes=read_floats(file.get(),uint64_t(maximum)*3);
        auto intrinsics=read_floats(file.get(),uint64_t(maximum)*9);
        auto features=read_floats(file.get(),uint64_t(maximum)*1024);
        auto angular=read_floats(file.get(),uint64_t(maximum)*6);
        std::vector<std::vector<float>> expected_motion,expected_camera;
        for(uint32_t length:lengths){
            expected_motion.push_back(read_floats(file.get(),uint64_t(length)*585));
            expected_camera.push_back(read_floats(file.get(),uint64_t(length)*3));
        }
        const char *backend_name=std::getenv("GEMX_TEST_BACKEND");
        if(!backend_name || !*backend_name)backend_name="CPU";
        const bool vulkan=std::strcmp(backend_name,"Vulkan")==0;
        gemx_session_config config{argv[1],argv[2],backend_name,nullptr,0,2,4};
        gemx_session *raw=nullptr;char message[512]{};
        auto status=gemx_session_create(&config,&raw,message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("create: ")+message);
        std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)> session(raw,gemx_session_destroy);
        for(size_t test=0;test<lengths.size();++test){
            uint32_t length=lengths[test];
            gemx_sequence_view input{length,keypoints.data(),boxes.data(),intrinsics.data(),
                                     features.data(),angular.data()};
            std::vector<float> motion(uint64_t(length)*585),camera(uint64_t(length)*3);
            status=gemx_infer(session.get(),&input,motion.data(),motion.size(),camera.data(),camera.size(),
                              message,sizeof(message));
            if(status!=GEMX_OK)throw std::runtime_error(std::string("infer: ")+message);
            auto motion_error=compare(motion.data(),expected_motion[test].data(),motion.size());
            auto camera_error=compare(camera.data(),expected_camera[test].data(),camera.size());
            std::printf("L=%u motion max=%g mean=%g camera max=%g mean=%g\n",length,
                        motion_error.maximum,motion_error.mean,camera_error.maximum,camera_error.mean);
            if(motion_error.maximum>2e-3f || motion_error.mean>1e-4 ||
               camera_error.maximum>2e-3f || camera_error.mean>1e-4)
                throw std::runtime_error("native output exceeds reference tolerance");
        }
        // Long offline sequences are returned in full as consecutive released
        // context windows; verify the boundary against independent calls.
        constexpr uint32_t long_frames=137;
        auto extend=[&](const std::vector<float> &source,uint32_t width){
            std::vector<float> value(uint64_t(long_frames)*width);
            std::copy_n(source.begin(),uint64_t(maximum)*width,value.begin());
            std::copy_n(source.begin(),uint64_t(long_frames-maximum)*width,
                        value.begin()+uint64_t(maximum)*width);return value;};
        auto long_keypoints=extend(keypoints,77*3),long_boxes=extend(boxes,3);
        auto long_intrinsics=extend(intrinsics,9),long_features=extend(features,1024);
        auto long_angular=extend(angular,6);
        gemx_sequence_view first_input{120,long_keypoints.data(),long_boxes.data(),long_intrinsics.data(),
                                      long_features.data(),long_angular.data()};
        gemx_sequence_view tail_input{17,long_keypoints.data()+120*77*3,long_boxes.data()+120*3,
            long_intrinsics.data()+120*9,long_features.data()+120*1024,long_angular.data()+120*6};
        gemx_sequence_view long_input{long_frames,long_keypoints.data(),long_boxes.data(),long_intrinsics.data(),
                                     long_features.data(),long_angular.data()};
        std::vector<float> first_motion(120*585),first_camera(120*3),tail_motion(17*585),tail_camera(17*3);
        std::vector<float> long_motion(uint64_t(long_frames)*585),long_camera(uint64_t(long_frames)*3);
        status=gemx_infer(session.get(),&first_input,first_motion.data(),first_motion.size(),
                          first_camera.data(),first_camera.size(),message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("first window: ")+message);
        status=gemx_infer(session.get(),&tail_input,tail_motion.data(),tail_motion.size(),
                          tail_camera.data(),tail_camera.size(),message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("tail window: ")+message);
        status=gemx_infer(session.get(),&long_input,long_motion.data(),long_motion.size(),
                          long_camera.data(),long_camera.size(),message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("long inference: ")+message);
        if(compare(long_motion.data(),first_motion.data(),first_motion.size()).maximum>1e-7f ||
           compare(long_motion.data()+120*585,tail_motion.data(),tail_motion.size()).maximum>1e-7f ||
           compare(long_camera.data(),first_camera.data(),first_camera.size()).maximum>1e-7f ||
           compare(long_camera.data()+120*3,tail_camera.data(),tail_camera.size()).maximum>1e-7f)
            throw std::runtime_error("long inference window boundary differs");
        const uint32_t length=30;
        gemx_sequence_view input{length,keypoints.data(),boxes.data(),intrinsics.data(),
                                 features.data(),angular.data()};
        std::vector<float> body(uint64_t(length)*76*3),identity(uint64_t(length)*45);
        std::vector<float> scales(uint64_t(length)*69),orient_camera(uint64_t(length)*3);
        std::vector<float> translation_camera(uint64_t(length)*3),orient_world(uint64_t(length)*3);
        std::vector<float> translation_world(uint64_t(length)*3);
        gemx_motion_view decoded{length,body.data(),identity.data(),scales.data(),orient_camera.data(),
                                 translation_camera.data(),orient_world.data(),translation_world.data()};
        status=gemx_infer_motion(session.get(),&input,&decoded,message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("infer motion: ")+message);
        require_finite(body,"body pose");require_finite(identity,"identity coefficients");
        require_finite(scales,"scale parameters");require_finite(orient_camera,"camera orientation");
        require_finite(translation_camera,"camera translation");require_finite(orient_world,"world orientation");
        require_finite(translation_world,"world translation");
        for(uint32_t frame=0;frame<length;++frame)
            if(scales[uint64_t(frame)*69]<.7f || scales[uint64_t(frame)*69]>1.f)
                throw std::runtime_error("global scale is outside the published clamp");
        if(argc>=5){
            file_ptr post_file(std::fopen(argv[4],"rb"),std::fclose);
            if(!post_file)throw std::runtime_error("cannot open postprocess fixture");
            char post_magic[8];read(post_file.get(),post_magic,8);
            uint32_t post_header[2];read(post_file.get(),post_header,sizeof(post_header));
            if(std::memcmp(post_magic,"GEMPOST1",8) || post_header[0]!=1 || post_header[1]!=length)
                throw std::runtime_error("unsupported postprocess fixture");
            std::array<std::pair<std::vector<float> *,const char *>,7> values{{
                {&body,"body pose"},{&identity,"identity coefficients"},{&scales,"scale parameters"},
                {&orient_camera,"camera orientation"},{&translation_camera,"camera translation"},
                {&orient_world,"world orientation"},{&translation_world,"world translation"}}};
            for(auto [actual,label]:values){
                auto expected=read_floats(post_file.get(),actual->size());
                auto error=compare(actual->data(),expected.data(),actual->size());
                std::printf("%s max=%g mean=%g\n",label,error.maximum,error.mean);
                if(error.maximum>(vulkan?3e-3f:2e-4f) || error.mean>(vulkan?5e-4:2e-5))
                    throw std::runtime_error(std::string(label)+" exceeds upstream tolerance");
            }
        }
        std::vector<float> positions(uint64_t(length)*77*3),rotations(uint64_t(length)*77*4);
        std::vector<float> local_translations(uint64_t(length)*77*3);
        std::vector<int32_t> parents(77);
        gemx_skeleton_view skeleton{length,positions.data(),rotations.data(),parents.data(),local_translations.data()};
        status=gemx_build_skeleton(session.get(),&decoded,&skeleton,message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("build skeleton: ")+message);
        require_finite(positions,"skeleton positions");require_finite(rotations,"skeleton rotations");
        if(parents[0]!=-1)throw std::runtime_error("SOMA Hips must be the root");
        for(uint32_t joint=1;joint<77;++joint)
            if(parents[joint]<0 || parents[joint]>=static_cast<int32_t>(joint))
                throw std::runtime_error("SOMA parent hierarchy is not topological");
        if(argc==6){
            file_ptr skeleton_file(std::fopen(argv[5],"rb"),std::fclose);
            if(!skeleton_file)throw std::runtime_error("cannot open skeleton fixture");
            char skeleton_magic[8];read(skeleton_file.get(),skeleton_magic,8);
            uint32_t skeleton_header[3];read(skeleton_file.get(),skeleton_header,sizeof(skeleton_header));
            if(std::memcmp(skeleton_magic,"GEMSKEL1",8) || skeleton_header[0]!=1 ||
               skeleton_header[1]!=length || skeleton_header[2]!=77)
                throw std::runtime_error("unsupported skeleton fixture");
            auto expected=read_floats(skeleton_file.get(),positions.size());
            auto error=compare(positions.data(),expected.data(),positions.size());
            std::printf("SOMA/MHR skeleton max=%g m mean=%g m\n",error.maximum,error.mean);
            if(error.maximum>(vulkan?3e-3f:2e-4f) || error.mean>(vulkan?5e-4:2e-5))
                throw std::runtime_error("SOMA/MHR skeleton exceeds upstream tolerance");
        }
        const std::filesystem::path glb=std::filesystem::temp_directory_path()/"gemx-reference.glb";
        status=gemx_export_skeleton_glb(session.get(),&decoded,30.f,glb.c_str(),message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("export skeleton: ")+message);
        file_ptr glb_file(std::fopen(glb.c_str(),"rb"),std::fclose);
        uint32_t glb_header[3]{};read(glb_file.get(),glb_header,sizeof(glb_header));
        if(glb_header[0]!=0x46546c67 || glb_header[1]!=2 ||
           glb_header[2]!=std::filesystem::file_size(glb))
            throw std::runtime_error("invalid exported GLB header");
        std::filesystem::remove(glb);
        status=gemx_export_skeleton_samples_glb(session.get(),&skeleton,30.f,glb.c_str(),message,sizeof(message));
        if(status!=GEMX_OK)throw std::runtime_error(std::string("export skeleton samples: ")+message);
        std::filesystem::remove(glb);
        gemx_profile profile{};
        status=gemx_session_get_profile(session.get(),&profile,message,sizeof(message));
        std::printf("profile calls=%llu frames=%llu cache_hits=%llu\n",
                    static_cast<unsigned long long>(profile.calls),
                    static_cast<unsigned long long>(profile.frames),
                    static_cast<unsigned long long>(profile.graph_cache_hits));
        // Reference lengths, three long-window checks and one decoded inference.
        if(status!=GEMX_OK || profile.calls!=lengths.size()+4 || profile.frames!=473 ||
           profile.graph_cache_hits<1)
            throw std::runtime_error("inference profile counters are inconsistent");
        std::printf("decoded motion, SOMA-77 forward kinematics and GLB export passed\n");
        return 0;
    }catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
