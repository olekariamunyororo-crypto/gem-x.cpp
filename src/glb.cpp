#include "skeleton.hpp"
#include "internal.hpp"
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace gemx { namespace {
void append_u32(std::vector<unsigned char> &out,uint32_t value){
    for(int i=0;i<4;++i)out.push_back(static_cast<unsigned char>(value>>(8*i)));
}
void append_float(std::vector<unsigned char> &out,float value){append_u32(out,std::bit_cast<uint32_t>(value));}
std::string quote(const std::string &value){
    std::ostringstream out;out<<'"';
    for(unsigned char c:value){
        if(c=='"' || c=='\\')out<<'\\'<<char(c);
        else if(c>=0x20)out<<char(c);
        else{const char *hex="0123456789abcdef";out<<"\\u00"<<hex[c>>4]<<hex[c&15];}
    }
    out<<'"';return out.str();
}
struct view {uint32_t offset,length;};
}

void write_glb(const skeleton_data &data,float fps,const std::string &path){
    require(data.frames>=1 && data.names.size()==77 && data.parents.size()==77 &&
            data.local_rotations.size()==uint64_t(data.frames)*77*4 &&
            data.local_translations.size()==uint64_t(data.frames)*77*3,
            "invalid skeleton animation for GLB export");
    std::vector<unsigned char> binary;std::vector<view> views;
    auto begin_view=[&](){while(binary.size()%4)binary.push_back(0);return static_cast<uint32_t>(binary.size());};
    auto finish_view=[&](uint32_t start){views.push_back({start,static_cast<uint32_t>(binary.size()-start)});};
    uint32_t start=begin_view();for(uint32_t frame=0;frame<data.frames;++frame)append_float(binary,frame/fps);
    finish_view(start);const uint32_t time_view=0;
    std::vector<uint32_t> rotation_views(77);
    for(uint32_t joint=0;joint<77;++joint){
        start=begin_view();rotation_views[joint]=views.size();
        for(uint32_t frame=0;frame<data.frames;++frame)for(int component=0;component<4;++component)
            append_float(binary,data.local_rotations[(uint64_t(frame)*77+joint)*4+component]);
        finish_view(start);
    }
    start=begin_view();const uint32_t translation_view=views.size();
    for(uint32_t frame=0;frame<data.frames;++frame)for(int component=0;component<3;++component)
        append_float(binary,data.local_translations[uint64_t(frame)*77*3+component]);
    finish_view(start);while(binary.size()%4)binary.push_back(0);

    std::ostringstream json;json<<"{\"asset\":{\"version\":\"2.0\",\"generator\":\"gem-x.cpp\","
        "\"copyright\":\"GEM-X and SOMA by NVIDIA; native port under Apache-2.0\"},";
    json<<"\"scene\":0,\"scenes\":[{\"nodes\":[0]}],\"nodes\":[";
    for(uint32_t joint=0;joint<77;++joint){
        if(joint)json<<',';
        json<<"{\"name\":"<<quote(data.names[joint]);
        json<<",\"translation\":[";
        for(int c=0;c<3;++c){if(c)json<<',';json<<data.local_translations[uint64_t(joint)*3+c];}
        json<<"],\"rotation\":[";
        for(int c=0;c<4;++c){if(c)json<<',';json<<data.local_rotations[uint64_t(joint)*4+c];}
        json<<']';bool first=true;
        for(uint32_t child=0;child<77;++child)if(data.parents[child]==static_cast<int32_t>(joint)){
            if(first){json<<",\"children\":[";first=false;}else json<<',';json<<child;
        }
        if(!first)json<<']';
        json<<'}';
    }
    json<<"],\"buffers\":[{\"byteLength\":"<<binary.size()<<"}],\"bufferViews\":[";
    for(size_t i=0;i<views.size();++i){if(i)json<<',';json<<"{\"buffer\":0,\"byteOffset\":"
        <<views[i].offset<<",\"byteLength\":"<<views[i].length<<'}';}
    json<<"],\"accessors\":[{\"bufferView\":"<<time_view<<",\"componentType\":5126,\"count\":"
        <<data.frames<<",\"type\":\"SCALAR\",\"min\":[0],\"max\":["<<(data.frames-1)/fps<<"]}";
    for(uint32_t joint=0;joint<77;++joint)json<<",{\"bufferView\":"<<rotation_views[joint]
        <<",\"componentType\":5126,\"count\":"<<data.frames<<",\"type\":\"VEC4\"}";
    const uint32_t translation_accessor=78;
    json<<",{\"bufferView\":"<<translation_view<<",\"componentType\":5126,\"count\":"
        <<data.frames<<",\"type\":\"VEC3\"}],\"animations\":[{\"name\":\"GEM-X SOMA motion\","
        "\"samplers\":[";
    for(uint32_t joint=0;joint<77;++joint){if(joint)json<<',';json<<"{\"input\":0,\"output\":"
        <<joint+1<<",\"interpolation\":\"LINEAR\"}";}
    json<<",{\"input\":0,\"output\":"<<translation_accessor<<",\"interpolation\":\"LINEAR\"}],"
        "\"channels\":[";
    for(uint32_t joint=0;joint<77;++joint){if(joint)json<<',';json<<"{\"sampler\":"<<joint
        <<",\"target\":{\"node\":"<<joint<<",\"path\":\"rotation\"}}";}
    json<<",{\"sampler\":77,\"target\":{\"node\":0,\"path\":\"translation\"}}]}],"
        "\"extras\":{\"source\":\"NVIDIA GEM-X / SOMA\",\"skeletonOnly\":true}}";
    std::string json_bytes=json.str();while(json_bytes.size()%4)json_bytes.push_back(' ');
    require(json_bytes.size()<=UINT32_MAX && binary.size()<=UINT32_MAX,"GLB exceeds 32-bit size limit");
    std::vector<unsigned char> output;output.reserve(12+8+json_bytes.size()+8+binary.size());
    append_u32(output,0x46546c67);append_u32(output,2);
    append_u32(output,static_cast<uint32_t>(12+8+json_bytes.size()+8+binary.size()));
    append_u32(output,json_bytes.size());append_u32(output,0x4e4f534a);
    output.insert(output.end(),json_bytes.begin(),json_bytes.end());
    append_u32(output,binary.size());append_u32(output,0x004e4942);output.insert(output.end(),binary.begin(),binary.end());
    std::filesystem::path destination(path);if(!destination.parent_path().empty())
        std::filesystem::create_directories(destination.parent_path());
    auto temporary=destination;temporary+=".tmp";
    {std::ofstream stream(temporary,std::ios::binary|std::ios::trunc);
     require(bool(stream),"cannot create temporary GLB");stream.write(reinterpret_cast<const char *>(output.data()),output.size());
     require(bool(stream),"cannot write complete GLB");}
    std::filesystem::rename(temporary,destination);
}
}
