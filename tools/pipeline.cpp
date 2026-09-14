#include "gemx.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using session_ptr=std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)>;
using vitpose_ptr=std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)>;
struct live_delete {void operator()(gemx_live *value) const{gemx_live_destroy(value);}};
using live_ptr=std::unique_ptr<gemx_live,live_delete>;
template<class T> void read(std::istream &in,T *value,size_t count=1){
    if(!in.read(reinterpret_cast<char *>(value),sizeof(T)*count))throw std::invalid_argument("truncated input");
}
template<class T> void write(std::ostream &out,const T *value,size_t count=1){
    if(!out.write(reinterpret_cast<const char *>(value),sizeof(T)*count))throw std::runtime_error("output write failed");
}
uint32_t number(const char *text){
    uint32_t value=0;auto end=text+std::strlen(text);auto result=std::from_chars(text,end,value);
    if(result.ec!=std::errc{}||result.ptr!=end)throw std::invalid_argument("invalid integer");
    return value;
}
float real(const char *text){
    char *end=nullptr;errno=0;float value=std::strtof(text,&end);
    if(errno||end!=text+std::strlen(text)||!std::isfinite(value))throw std::invalid_argument("invalid number");
    return value;
}
void api(gemx_status status,const char *message){if(status!=GEMX_OK)throw std::runtime_error(message);}

struct image_data {
    uint32_t width=0,height=0,stride=0;
    std::array<float,4> box{},camera{};
    std::vector<uint8_t> rgb;
};
image_data load_image(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open packed image");
    std::array<char,8> magic{};read(in,magic.data(),magic.size());
    if(std::string(magic.data(),magic.size())!="S3DIMG01")throw std::invalid_argument("wrong packed image magic");
    image_data value;read(in,&value.width);read(in,&value.height);read(in,&value.stride);
    read(in,value.box.data(),4);read(in,value.camera.data(),4);
    if(value.width<8||value.height<8||value.width>32766||value.height>32766||
       uint64_t(value.width)*value.height>16000000||value.stride!=value.width*3)
        throw std::invalid_argument("invalid packed image dimensions");
    value.rgb.resize(uint64_t(value.stride)*value.height);read(in,value.rgb.data(),value.rgb.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing packed image data");
    return value;
}

std::array<float,1024> load_pose_token(const std::string &path){
    for(unsigned attempt=0;attempt<600000;++attempt){
        std::error_code error;
        if(std::filesystem::file_size(path,error)>=400000&&!error)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("body result did not become available");
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!="S3DOUT01")throw std::invalid_argument("wrong body result magic");
    uint32_t tensors=0;read(in,&tensors);if(tensors<1||tensors>64)throw std::invalid_argument("invalid body tensor count");
    std::array<float,1024> result{};
    for(uint32_t tensor=0;tensor<tensors;++tensor){
        uint32_t name_size=0,type=0,rank=0;uint64_t elements=0;read(in,&name_size);
        if(name_size<1||name_size>64)throw std::invalid_argument("invalid tensor name");
        std::string name(name_size,'\0');read(in,name.data(),name.size());
        read(in,&type);read(in,&rank);read(in,&elements);
        if(rank>8||elements>(1ull<<28))throw std::invalid_argument("invalid tensor descriptor");
        uint64_t product=1;std::array<uint64_t,8> dimensions{};
        for(uint32_t i=0;i<rank;++i){read(in,&dimensions[i]);if(!dimensions[i]||dimensions[i]>(1ull<<28)/product)throw std::invalid_argument("invalid tensor shape");product*=dimensions[i];}
        if(product!=elements)throw std::invalid_argument("tensor element count mismatch");
        if(name=="pose_token"){
            if(type!=1||rank!=2||dimensions[0]!=1||dimensions[1]!=1024)
                throw std::invalid_argument("invalid GEM-X pose token");
            read(in,result.data(),result.size());return result;
        }else{
            in.seekg(static_cast<std::streamoff>(elements*4),std::ios::cur);
            if(!in)throw std::invalid_argument("truncated tensor payload");
        }
    }
    throw std::invalid_argument("body result lacks GEM-X pose token");
}

struct pose_sample {
    std::array<float,77*3> positions{},local_translations{},keypoints{};
    std::array<float,77*4> rotations{};
    std::array<int32_t,77> parents{};
    std::array<float,3> camera{};
};
void save_pose(const std::string &path,const pose_sample &value){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create GEM-X pose");
    write(out,"GEMPOSE1",8);write(out,value.positions.data(),value.positions.size());
    write(out,value.rotations.data(),value.rotations.size());
    write(out,value.local_translations.data(),value.local_translations.size());
    write(out,value.parents.data(),value.parents.size());write(out,value.camera.data(),3);
    write(out,value.keypoints.data(),value.keypoints.size());
}
pose_sample load_pose(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open GEM-X pose");
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!="GEMPOSE1")throw std::invalid_argument("wrong GEM-X pose magic");
    pose_sample value;read(in,value.positions.data(),value.positions.size());
    read(in,value.rotations.data(),value.rotations.size());
    read(in,value.local_translations.data(),value.local_translations.size());
    read(in,value.parents.data(),value.parents.size());read(in,value.camera.data(),3);
    read(in,value.keypoints.data(),value.keypoints.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing GEM-X pose data");
    return value;
}

std::string field(bool boundary=false){
    std::array<uint8_t,4> bytes{};size_t got=std::fread(bytes.data(),1,4,stdin);
    if(!got&&boundary&&std::feof(stdin))return {};
    if(got!=4)throw std::invalid_argument("truncated worker request");
    uint32_t size=uint32_t(bytes[0])|uint32_t(bytes[1])<<8|uint32_t(bytes[2])<<16|uint32_t(bytes[3])<<24;
    if(!size||size>4096)throw std::invalid_argument("invalid worker field");
    std::string value(size,'\0');if(std::fread(value.data(),1,size,stdin)!=size||value.find('\0')!=std::string::npos)
        throw std::invalid_argument("invalid worker field data");
    return value;
}
gemx_session_config config(const char *model,const char *module,const char *backend,
                           const char *description,uint32_t device,uint32_t threads,uint32_t cache){
    return {model,module,backend,std::strcmp(description,"-")?description:"",device,threads,cache};
}

int worker(int argc,char **argv){
    if(argc!=10)throw std::invalid_argument("usage: gemx-pipeline --worker DENOISER VITPOSE MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS CONTEXT");
    const uint32_t device=number(argv[6]),threads=number(argv[8]),context=number(argv[9]);
    char error[512]{};
    auto den_cfg=config(argv[2],argv[4],argv[5],argv[7],device,threads,8);
    gemx_session *raw_session=nullptr;api(gemx_session_create(&den_cfg,&raw_session,error,sizeof(error)),error);
    session_ptr session(raw_session,gemx_session_destroy);
    auto pose_cfg=config(argv[3],argv[4],argv[5],argv[7],device,threads,4);
    gemx_vitpose *raw_pose=nullptr;api(gemx_vitpose_create(&pose_cfg,&raw_pose,error,sizeof(error)),error);
    vitpose_ptr pose(raw_pose,gemx_vitpose_destroy);
    gemx_live *raw_live=nullptr;api(gemx_live_create(session.get(),context,&raw_live,error,sizeof(error)),error);
    live_ptr live(raw_live);std::string current_stream;
    std::fputs("READY\n",stdout);std::fflush(stdout);
    for(;;){
        auto stream=field(true);if(stream.empty())break;
        if(stream.size()>128)throw std::invalid_argument("stream identifier is too long");
        auto image_path=field(),body_path=field(),output_path=field();
        if(stream!=current_stream){gemx_live_reset(live.get());current_stream=stream;}
        image_data image;
        try{image=load_image(image_path);}catch(const std::exception &e){throw std::runtime_error(std::string("packed image: ")+e.what());}
        const float width=image.box[2]-image.box[0],height=image.box[3]-image.box[1];
        const float size=std::max(height,width/.75f)*1.2f;
        gemx_rgb_frame frame{image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,
                             {(image.box[0]+image.box[2])*.5f,(image.box[1]+image.box[3])*.5f,size}};
        pose_sample sample;api(gemx_vitpose_infer_rgb(pose.get(),&frame,1,sample.keypoints.data(),
                                                     sample.keypoints.size(),error,sizeof(error)),error);
        std::array<float,1024> token{};
        try{token=load_pose_token(body_path);}catch(const std::exception &e){throw std::runtime_error(std::string("body token: ")+e.what());}
        std::array<float,9> K={image.camera[0],0,image.camera[2],0,image.camera[1],image.camera[3],0,0,1};
        std::array<float,3> box={frame.box[0],frame.box[1],frame.box[2]};
        std::array<float,6> angular{};
        gemx_sequence_view observation{1,sample.keypoints.data(),box.data(),K.data(),token.data(),angular.data()};
        std::array<float,585> pred{};std::array<float,3> pred_camera{};
        api(gemx_live_push(live.get(),&observation,pred.data(),pred_camera.data(),error,sizeof(error)),error);
        std::array<float,76*3> body{};std::array<float,45> identity{};std::array<float,69> scales{};
        std::array<float,3> orient_camera{},translation_camera{},orient_world{},translation_world{};
        gemx_motion_view motion{1,body.data(),identity.data(),scales.data(),orient_camera.data(),
                                translation_camera.data(),orient_world.data(),translation_world.data()};
        api(gemx_decode_predictions(session.get(),&observation,pred.data(),pred.size(),pred_camera.data(),
                                    pred_camera.size(),&motion,error,sizeof(error)),error);
        gemx_skeleton_view skeleton{1,sample.positions.data(),sample.rotations.data(),sample.parents.data(),
                                    sample.local_translations.data()};
        api(gemx_build_skeleton(session.get(),&motion,&skeleton,error,sizeof(error)),error);
        sample.camera=translation_camera;save_pose(output_path,sample);
        std::fputs("DONE\n",stdout);std::fflush(stdout);
    }
    return 0;
}

int export_sequence(int argc,char **argv){
    if(argc!=11)throw std::invalid_argument("usage: gemx-pipeline --export DENOISER MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS FPS MANIFEST OUTPUT");
    char error[512]{};auto cfg=config(argv[2],argv[3],argv[4],argv[6],number(argv[5]),number(argv[7]),1);
    gemx_session *raw=nullptr;api(gemx_session_create(&cfg,&raw,error,sizeof(error)),error);
    session_ptr session(raw,gemx_session_destroy);
    std::ifstream manifest(argv[9],std::ios::binary);if(!manifest)throw std::runtime_error("cannot open manifest");
    std::array<char,8> magic{};read(manifest,magic.data(),8);
    if(std::string(magic.data(),8)!="GEMMAN01")throw std::invalid_argument("wrong manifest magic");
    uint32_t count=0;read(manifest,&count);if(count<1||count>GEMX_MAX_FRAMES)throw std::invalid_argument("invalid manifest count");
    std::vector<float> positions(uint64_t(count)*77*3),rotations(uint64_t(count)*77*4),translations(uint64_t(count)*77*3);
    std::array<int32_t,77> parents{};
    for(uint32_t i=0;i<count;++i){
        uint32_t size=0;read(manifest,&size);if(!size||size>4096)throw std::invalid_argument("invalid manifest path");
        std::string path(size,'\0');read(manifest,path.data(),size);if(path.find('\0')!=std::string::npos)throw std::invalid_argument("invalid manifest path data");
        auto sample=load_pose(path);
        std::copy(sample.positions.begin(),sample.positions.end(),positions.begin()+uint64_t(i)*77*3);
        std::copy(sample.rotations.begin(),sample.rotations.end(),rotations.begin()+uint64_t(i)*77*4);
        std::copy(sample.local_translations.begin(),sample.local_translations.end(),translations.begin()+uint64_t(i)*77*3);
        if(i==0)parents=sample.parents;else if(parents!=sample.parents)throw std::invalid_argument("SOMA topology changed");
    }
    gemx_skeleton_view skeleton{count,positions.data(),rotations.data(),parents.data(),translations.data()};
    api(gemx_export_skeleton_samples_glb(session.get(),&skeleton,real(argv[8]),argv[10],error,sizeof(error)),error);
    return 0;
}
}

int main(int argc,char **argv){
    try{
        if(argc>1&&!std::strcmp(argv[1],"--worker"))return worker(argc,argv);
        if(argc>1&&!std::strcmp(argv[1],"--export"))return export_sequence(argc,argv);
        throw std::invalid_argument("expected --worker or --export");
    }catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
