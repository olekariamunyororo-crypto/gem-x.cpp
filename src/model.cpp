#include "model.hpp"
#include "internal.hpp"
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

namespace gemx {
model::model(const std::string &path,ggml_backend_buffer_type_t buffer_type,
             const std::string &architecture,size_t expected_tensors){
  try{
    gguf_init_params params{true,&context_};
    gguf_=gguf_init_from_file(path.c_str(),params);
    require(gguf_!=nullptr,"cannot load GEM-X GGUF");
    require(string("general.architecture")==architecture,"GGUF has unexpected architecture");
    if(architecture=="gemx")
        require(u32("gemx.context_length")==120 && u32("gemx.embedding_length")==512 &&
                u32("gemx.feed_forward_length")==2048 && u32("gemx.block_count")==12 &&
                u32("gemx.attention.head_count")==8 && u32("gemx.motion_length")==585,
                "unsupported GEM-X model architecture");
    const size_t count=static_cast<size_t>(gguf_get_n_tensors(gguf_));
    require(count==expected_tensors || (architecture=="gemx" && (count==247 || count==251)),
            "unexpected GGUF tensor count");
    for(size_t i=0;i<count;++i){
        const char *name=gguf_get_tensor_name(gguf_,i);
        auto *value=ggml_get_tensor(context_,name);
        require(value!=nullptr,"GGUF tensor metadata is inconsistent");
        tensors_.emplace(name,value);
    }
    buffer_=ggml_backend_alloc_ctx_tensors_from_buft(context_,buffer_type);
    if(!buffer_)throw std::bad_alloc();
    ggml_backend_buffer_set_usage(buffer_,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    const size_t base=gguf_get_data_offset(gguf_);
    struct file_closer {void operator()(FILE *value) const noexcept{if(value)std::fclose(value);}};
    std::unique_ptr<FILE,file_closer> file(std::fopen(path.c_str(),"rb"));
    require(file!=nullptr,"cannot reopen GEM-X GGUF data");
    std::vector<unsigned char> bytes;
    for(size_t i=0;i<count;++i){
        const char *name=gguf_get_tensor_name(gguf_,i);
        auto *value=tensors_.at(name);
        const size_t size=ggml_nbytes(value);
        bytes.resize(size);
        require(std::fseek(file.get(),static_cast<long>(base+gguf_get_tensor_offset(gguf_,i)),SEEK_SET)==0 &&
                std::fread(bytes.data(),1,size,file.get())==size,"truncated GEM-X GGUF tensor data");
        ggml_backend_tensor_set(value,bytes.data(),0,size);
    }
  }catch(...){
    if(buffer_)ggml_backend_buffer_free(buffer_);
    if(context_)ggml_free(context_);
    if(gguf_)gguf_free(gguf_);
    buffer_=nullptr;context_=nullptr;gguf_=nullptr;
    throw;
  }
}

model::~model(){
    if(packed_buffer_)ggml_backend_buffer_free(packed_buffer_);
    if(packed_context_)ggml_free(packed_context_);
    if(buffer_)ggml_backend_buffer_free(buffer_);
    if(context_)ggml_free(context_);
    if(gguf_)gguf_free(gguf_);
}

// Opt-in load-time packing. Original weights remain owned by the model;
// no concatenation or extra weight transfers occur during inference.
void model::pack_vitpose_gate_up(ggml_backend_buffer_type_t buffer_type){
    require(!packed_context_,"ViTPose weights already packed");
    packed_context_=ggml_init({1024*1024,nullptr,true});
    if(!packed_context_)throw std::bad_alloc();
    for(int block=0;block<32;++block){
        const std::string p="backbone.block."+std::to_string(block)+".mlp.";
        for(const auto *suffix:{"weight","bias"}){
            auto *a=tensor(p+"w1."+suffix),*b=tensor(p+"w2."+suffix);
            require(a->type==GGML_TYPE_F32 && b->type==GGML_TYPE_F32 && ggml_are_same_shape(a,b),"unsupported packed FFN weights");
            auto *combined=std::strcmp(suffix,"weight")==0 ?
                ggml_new_tensor_2d(packed_context_,GGML_TYPE_F32,a->ne[0],a->ne[1]*2) :
                ggml_new_tensor_1d(packed_context_,GGML_TYPE_F32,a->ne[0]*2);
            const std::string name=p+"w12."+suffix;ggml_set_name(combined,name.c_str());tensors_.emplace(name,combined);
        }
    }
    packed_buffer_=ggml_backend_alloc_ctx_tensors_from_buft(packed_context_,buffer_type);
    if(!packed_buffer_)throw std::bad_alloc();
    ggml_backend_buffer_set_usage(packed_buffer_,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for(int block=0;block<32;++block){
        const std::string p="backbone.block."+std::to_string(block)+".mlp.";
        for(const auto *suffix:{"weight","bias"}){
            auto *a=tensor(p+"w1."+suffix),*b=tensor(p+"w2."+suffix),*combined=tensor(p+"w12."+suffix);
            const size_t bytes=ggml_nbytes(a);std::vector<unsigned char> host(bytes);
            ggml_backend_tensor_get(a,host.data(),0,bytes);ggml_backend_tensor_set(combined,host.data(),0,bytes);
            ggml_backend_tensor_get(b,host.data(),0,bytes);ggml_backend_tensor_set(combined,host.data(),bytes,bytes);
        }
    }
}

ggml_tensor *model::tensor(const std::string &name) const{
    auto found=tensors_.find(name);
    if(found==tensors_.end())throw std::invalid_argument("missing GEM-X tensor: "+name);
    return found->second;
}

uint32_t model::u32(const char *name) const{
    int index=gguf_find_key(gguf_,name);require(index>=0,"missing GEM-X metadata");
    require(gguf_get_kv_type(gguf_,index)==GGUF_TYPE_UINT32,"invalid GEM-X integer metadata");
    return gguf_get_val_u32(gguf_,index);
}

float model::f32(const char *name) const{
    int index=gguf_find_key(gguf_,name);require(index>=0,"missing GGUF metadata");
    require(gguf_get_kv_type(gguf_,index)==GGUF_TYPE_FLOAT32,"invalid GGUF float metadata");
    return gguf_get_val_f32(gguf_,index);
}

std::string model::string(const char *name) const{
    int index=gguf_find_key(gguf_,name);require(index>=0,"missing GEM-X metadata");
    require(gguf_get_kv_type(gguf_,index)==GGUF_TYPE_STRING,"invalid GEM-X string metadata");
    return gguf_get_val_str(gguf_,index);
}

std::vector<float> model::read_f32(const std::string &name) const{
    auto *value=tensor(name);require(value->type==GGML_TYPE_F32,"expected F32 GEM-X tensor");
    std::vector<float> result(ggml_nelements(value));
    ggml_backend_tensor_get(value,result.data(),0,result.size()*sizeof(float));
    return result;
}

std::vector<int32_t> model::read_i32(const std::string &name) const{
    auto *value=tensor(name);require(value->type==GGML_TYPE_I32,"expected I32 GEM-X tensor");
    std::vector<int32_t> result(ggml_nelements(value));
    ggml_backend_tensor_get(value,result.data(),0,result.size()*sizeof(int32_t));
    return result;
}

std::vector<std::string> model::string_array(const char *name) const{
    int index=gguf_find_key(gguf_,name);require(index>=0,"missing GEM-X array metadata");
    require(gguf_get_kv_type(gguf_,index)==GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(gguf_,index)==GGUF_TYPE_STRING,"invalid GEM-X string-array metadata");
    std::vector<std::string> result;
    const size_t count=static_cast<size_t>(gguf_get_arr_n(gguf_,index));
    for(size_t i=0;i<count;++i)result.emplace_back(gguf_get_arr_str(gguf_,index,i));
    return result;
}
}
