#include "session.hpp"
#include "internal.hpp"
#include "ggml-alloc.h"
#include <array>
#include <chrono>
#include <cstring>

namespace gemx { namespace {
using clock_type=std::chrono::steady_clock;
uint64_t elapsed(clock_type::time_point start){
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock_type::now()-start).count());
}

ggml_tensor *linear(ggml_context *ctx,const model &weights,const std::string &prefix,
                    ggml_tensor *input){
    auto *result=ggml_mul_mat(ctx,weights.tensor(prefix+".weight"),input);
    ggml_mul_mat_set_prec(result,GGML_PREC_F32);
    return ggml_add(ctx,result,weights.tensor(prefix+".bias"));
}

ggml_tensor *layer_norm(ggml_context *ctx,const model &weights,const std::string &prefix,
                        ggml_tensor *input,float epsilon){
    auto *result=ggml_norm(ctx,input,epsilon);
    result=ggml_mul(ctx,result,weights.tensor(prefix+".weight"));
    return ggml_add(ctx,result,weights.tensor(prefix+".bias"));
}

ggml_tensor *gelu_tanh(ggml_context *ctx,ggml_tensor *input){
    auto *cube=ggml_mul(ctx,ggml_sqr(ctx,input),input);
    auto *inside=ggml_add(ctx,input,ggml_scale(ctx,cube,0.044715f));
    inside=ggml_tanh(ctx,ggml_scale(ctx,inside,0.7978845608028654f));
    inside=ggml_scale_bias(ctx,inside,0.5f,0.5f);
    return ggml_mul(ctx,input,inside);
}

constexpr std::array<uint32_t,33> joints={0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,
                                           18,28,39,40,41,42,46,56,67,68,69,70,
                                           71,72,73,74,75,76};
}

struct session::graph_state {
    ggml_context *context=nullptr;
    ggml_cgraph *graph=nullptr;
    ggml_gallocr_t allocator=nullptr;
    ggml_tensor *xy=nullptr,*visible=nullptr,*cliff=nullptr,*image=nullptr,*angular=nullptr;
    ggml_tensor *positions=nullptr,*output=nullptr;
    uint32_t frames=0;
    std::list<uint32_t>::iterator lru;
    ~graph_state(){
        if(allocator)ggml_gallocr_free(allocator);
        if(context)ggml_free(context);
    }
};

session::~session()=default;

session::session(const gemx_session_config &config){
    require(config.model_path && config.backend_module && config.backend_name,
            "model, backend module and backend name are required");
    require(config.threads>=1 && config.threads<=1024,"threads must be in 1..1024");
    cache_capacity_=config.graph_cache_capacity?config.graph_cache_capacity:4;
    require(cache_capacity_<=32,"graph cache capacity must be at most 32");
    backend_=std::make_unique<backend>(config.backend_module,config.backend_name,
        config.device_index,config.threads,
        config.expected_device_description?config.expected_device_description:"");
    model_=std::make_unique<model>(config.model_path,backend_->buffer_type());
    motion_mean_=model_->read_f32("motion.mean");motion_std_=model_->read_f32("motion.std");
    soma_local_=model_->read_f32("soma.rest_local");soma_world_=model_->read_f32("soma.rest_world");
    soma_parents_=model_->read_i32("soma.parents");soma_names_=model_->string_array("gemx.soma_joint_names");
    auto &id=soma_identity_;
    id.mhr_offsets=model_->read_f32("mhr.offsets");
    id.mhr_prerotations=model_->read_f32("mhr.prerotations");
    id.mhr_parents=model_->read_i32("mhr.parents");
    id.mhr_parameter_matrix=model_->read_f32("mhr.parameter_matrix");
    id.mhr_inverse_bind=model_->read_f32("mhr.inverse_bind");
    id.mhr_skin_joints=model_->read_i32("mhr.skin_joints");
    id.mhr_skin_weights=model_->read_f32("mhr.skin_weights");
    id.mhr_skin_vertices=model_->read_i32("mhr.skin_vertices");
    id.mhr_shape_vectors=model_->read_f32("mhr.shape_vectors");
    id.mhr_base_shape=model_->read_f32("mhr.base_shape");
    id.mhr_faces=model_->read_i32("mhr.faces");
    id.transfer_face_ids=model_->read_i32("soma.transfer_face_ids");
    id.transfer_barycentric=model_->read_f32("soma.transfer_barycentric");
    id.rbf_crow=model_->read_i32("soma.rbf_crow");
    id.rbf_columns=model_->read_i32("soma.rbf_columns");
    id.rbf_values=model_->read_f32("soma.rbf_values");
    id.bind_world=model_->read_f32("soma.bind_world");
    id.soma_parents=soma_parents_;
    id.rotation_crow=model_->read_i32("soma.rotation_crow");
    id.rotation_vertices=model_->read_i32("soma.rotation_vertices");
    id.rotation_reference=model_->read_f32("soma.rotation_reference");
}

session::graph_state &session::graph(uint32_t frames){
    const uint32_t cache_key=frames;
    if(auto found=graphs_.find(cache_key);found!=graphs_.end()){
        lru_.erase(found->second->lru);lru_.push_front(cache_key);found->second->lru=lru_.begin();
        ++profile_.graph_cache_hits;return *found->second;
    }
    const auto started=clock_type::now();
    auto state=std::make_unique<graph_state>();state->frames=frames;
    state->context=ggml_init({8u*1024u*1024u,nullptr,true});
    if(!state->context)throw std::bad_alloc();
    auto *ctx=state->context;
    state->graph=ggml_new_graph_custom(ctx,2048,false);
    if(!state->graph)throw std::bad_alloc();
    state->xy=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,2,33,frames);
    state->visible=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,1,33,frames);
    state->cliff=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,3,frames);
    state->image=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1024,frames);
    state->angular=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,6,frames);
    state->positions=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,frames);
    for(auto [tensor,name]:std::array<std::pair<ggml_tensor *,const char *>,6>{{
        {state->xy,"input.xy"},{state->visible,"input.visible"},{state->cliff,"input.cliff"},
        {state->image,"input.image"},{state->angular,"input.angular"},
        {state->positions,"input.positions"}}}){
        ggml_set_name(tensor,name);ggml_set_input(tensor);
    }
    auto *xy=state->xy,*visible=state->visible,*cliff_input=state->cliff;
    auto *image_input=state->image,*angular_input=state->angular;

    auto *obs=linear(ctx,*model_,"obs.xy",xy);
    auto *vis=ggml_repeat_4d(ctx,visible,32,33,frames,1);
    auto *missing=ggml_repeat_4d(ctx,model_->tensor("obs.missing"),32,33,frames,1);
    obs=ggml_add(ctx,ggml_mul(ctx,obs,vis),
        ggml_mul(ctx,missing,ggml_scale_bias(ctx,vis,-1.f,1.f)));
    obs=ggml_reshape_2d(ctx,obs,1056,frames);
    obs=linear(ctx,*model_,"obs.mlp.0",obs);
    obs=ggml_gelu_erf(ctx,obs);
    obs=linear(ctx,*model_,"obs.mlp.2",obs);

    auto *cliff=linear(ctx,*model_,"cliff.mlp.0",cliff_input);
    cliff=ggml_silu(ctx,cliff);cliff=linear(ctx,*model_,"cliff.mlp.2",cliff);
    cliff=linear(ctx,*model_,"cliff.exists.0",cliff);
    cliff=ggml_silu(ctx,cliff);cliff=linear(ctx,*model_,"cliff.exists.2",cliff);

    auto *image=layer_norm(ctx,*model_,"image.norm",image_input,1e-5f);
    image=linear(ctx,*model_,"image.proj",image);
    image=linear(ctx,*model_,"image.exists.0",image);
    image=ggml_silu(ctx,image);image=linear(ctx,*model_,"image.exists.2",image);

    auto *angular=ggml_div(ctx,ggml_sub(ctx,angular_input,model_->tensor("angular.mean")),
                           model_->tensor("angular.std"));
    angular=linear(ctx,*model_,"angular.mlp.0",angular);
    angular=ggml_silu(ctx,angular);angular=linear(ctx,*model_,"angular.mlp.2",angular);
    angular=linear(ctx,*model_,"angular.exists.0",angular);
    angular=ggml_silu(ctx,angular);angular=linear(ctx,*model_,"angular.exists.2",angular);

    auto *hidden=ggml_add(ctx,ggml_add(ctx,obs,cliff),ggml_add(ctx,image,angular));
    auto *time=linear(ctx,*model_,"time.mlp.0",model_->tensor("time.position"));
    time=ggml_silu(ctx,time);time=linear(ctx,*model_,"time.mlp.2",time);
    hidden=ggml_add(ctx,hidden,time);
    hidden=linear(ctx,*model_,"input",hidden);

    for(uint32_t block=0;block<12;++block){
        const std::string prefix="block."+std::to_string(block);
        auto *norm=layer_norm(ctx,*model_,prefix+".norm1",hidden,1e-6f);
        auto *qkv=linear(ctx,*model_,prefix+".attn.qkv",norm);
        auto *query=ggml_view_2d(ctx,qkv,512,frames,qkv->nb[1],0);
        auto *key=ggml_view_2d(ctx,qkv,512,frames,qkv->nb[1],512*sizeof(float));
        auto *value=ggml_view_2d(ctx,qkv,512,frames,qkv->nb[1],1024*sizeof(float));
        query=ggml_reshape_3d(ctx,ggml_cont(ctx,query),64,8,frames);
        key=ggml_reshape_3d(ctx,ggml_cont(ctx,key),64,8,frames);
        value=ggml_reshape_3d(ctx,ggml_cont(ctx,value),64,8,frames);
        query=ggml_rope(ctx,query,state->positions,64,GGML_ROPE_TYPE_NORMAL);
        key=ggml_rope(ctx,key,state->positions,64,GGML_ROPE_TYPE_NORMAL);
        query=ggml_cont(ctx,ggml_permute(ctx,query,0,2,1,3));
        key=ggml_cont(ctx,ggml_permute(ctx,key,0,2,1,3));
        value=ggml_cont(ctx,ggml_permute(ctx,value,1,2,0,3));
        auto *score=ggml_mul_mat(ctx,key,query);
        ggml_mul_mat_set_prec(score,GGML_PREC_F32);
        score=ggml_soft_max(ctx,ggml_scale(ctx,score,0.125f));
        auto *attention=ggml_mul_mat(ctx,value,score);
        ggml_mul_mat_set_prec(attention,GGML_PREC_F32);
        attention=ggml_cont(ctx,ggml_permute(ctx,attention,0,2,1,3));
        attention=ggml_reshape_2d(ctx,attention,512,frames);
        attention=linear(ctx,*model_,prefix+".attn.output",attention);
        hidden=ggml_add(ctx,hidden,ggml_mul(ctx,attention,model_->tensor(prefix+".attn.gate")));
        norm=layer_norm(ctx,*model_,prefix+".norm2",hidden,1e-6f);
        auto *mlp=linear(ctx,*model_,prefix+".mlp.0",norm);
        mlp=gelu_tanh(ctx,mlp);mlp=linear(ctx,*model_,prefix+".mlp.2",mlp);
        hidden=ggml_add(ctx,hidden,ggml_mul(ctx,mlp,model_->tensor(prefix+".mlp.gate")));
    }

    auto *motion=linear(ctx,*model_,"output.mlp.0",hidden);
    motion=ggml_gelu_erf(ctx,motion);motion=linear(ctx,*model_,"output.mlp.2",motion);
    auto *body=ggml_view_2d(ctx,motion,456,frames,motion->nb[1],0);
    auto *identity=ggml_view_2d(ctx,motion,45,frames,motion->nb[1],456*sizeof(float));
    identity=ggml_cont(ctx,ggml_transpose(ctx,identity));
    identity=ggml_sum_rows(ctx,identity);
    identity=ggml_scale(ctx,identity,1.f/static_cast<float>(frames));
    identity=ggml_reshape_1d(ctx,identity,45);
    identity=ggml_repeat_4d(ctx,identity,45,frames,1,1);
    auto *scale=ggml_view_2d(ctx,motion,28,frames,motion->nb[1],501*sizeof(float));
    scale=ggml_cont(ctx,ggml_transpose(ctx,scale));
    scale=ggml_sum_rows(ctx,scale);
    scale=ggml_scale(ctx,scale,1.f/static_cast<float>(frames));
    scale=ggml_reshape_1d(ctx,scale,28);
    scale=ggml_mul_mat(ctx,model_->tensor("output.scale_components"),scale);
    ggml_mul_mat_set_prec(scale,GGML_PREC_F32);
    scale=ggml_add(ctx,scale,model_->tensor("output.scale_mean"));
    scale=ggml_repeat_4d(ctx,scale,69,frames,1,1);
    auto *tail=ggml_view_2d(ctx,motion,15,frames,motion->nb[1],570*sizeof(float));
    motion=ggml_concat(ctx,ggml_concat(ctx,body,identity,0),ggml_concat(ctx,scale,tail,0),0);

    auto *camera=linear(ctx,*model_,"camera.mlp.0",hidden);
    camera=ggml_gelu_erf(ctx,camera);camera=linear(ctx,*model_,"camera.mlp.2",camera);
    camera=ggml_add(ctx,ggml_mul(ctx,camera,model_->tensor("camera.std")),
                    model_->tensor("camera.mean"));
    state->output=ggml_concat(ctx,motion,camera,0);
    ggml_set_name(state->output,"output.prediction");ggml_set_output(state->output);
    ggml_build_forward_expand(state->graph,state->output);
    state->allocator=ggml_gallocr_new(backend_->buffer_type());
    if(!state->allocator || !ggml_gallocr_reserve(state->allocator,state->graph) ||
       !ggml_gallocr_alloc_graph(state->allocator,state->graph))
        throw std::bad_alloc();
    lru_.push_front(cache_key);state->lru=lru_.begin();
    auto [position,inserted]=graphs_.emplace(cache_key,std::move(state));
    (void)inserted;
    while(graphs_.size()>cache_capacity_){auto key=lru_.back();lru_.pop_back();graphs_.erase(key);}
    ++profile_.graph_cache_misses;profile_.graph_build_ns+=elapsed(started);
    return *position->second;
}

void session::infer_window(const gemx_sequence_view &input,float *motion,float *camera){
    require(input.frames>=1 && input.frames<=120,"native denoiser window accepts 1..120 frames");
    const auto preprocessing_started=clock_type::now();
    std::vector<float> normalized(uint64_t(input.frames)*77*3),cliff(uint64_t(input.frames)*3);
    char message[256]{};
    auto status=gemx_preprocess_sequence(&input,normalized.data(),normalized.size(),
                                         cliff.data(),cliff.size(),message,sizeof(message));
    if(status!=GEMX_OK)throw std::invalid_argument(message);
    std::vector<float> xy(uint64_t(input.frames)*33*2),visible(uint64_t(input.frames)*33);
    for(uint32_t f=0;f<input.frames;++f)for(uint32_t j=0;j<33;++j){
        const auto *source=normalized.data()+(uint64_t(f)*77+joints[j])*3;
        xy[(uint64_t(f)*33+j)*2]=source[0];xy[(uint64_t(f)*33+j)*2+1]=source[1];
        visible[uint64_t(f)*33+j]=source[2]>.5f?1.f:0.f;
    }
    std::vector<int32_t> positions(input.frames);for(uint32_t i=0;i<input.frames;++i)positions[i]=i;
    profile_.preprocessing_ns+=elapsed(preprocessing_started);
    auto &state=graph(input.frames);
    const auto upload_started=clock_type::now();
    auto set=[&](ggml_tensor *tensor,const void *data,size_t bytes){
        ggml_backend_tensor_set(tensor,data,0,bytes);
    };
    set(state.xy,xy.data(),xy.size()*sizeof(float));
    set(state.visible,visible.data(),visible.size()*sizeof(float));
    set(state.cliff,cliff.data(),cliff.size()*sizeof(float));
    set(state.image,input.body_features,uint64_t(input.frames)*1024*sizeof(float));
    set(state.angular,input.camera_angular_velocity,uint64_t(input.frames)*6*sizeof(float));
    set(state.positions,positions.data(),positions.size()*sizeof(int32_t));
    profile_.upload_ns+=elapsed(upload_started);
    const auto inference_started=clock_type::now();backend_->compute(state.graph);
    profile_.inference_ns+=elapsed(inference_started);
    const auto download_started=clock_type::now();
    std::vector<float> output(uint64_t(input.frames)*588);
    ggml_backend_tensor_get(state.output,output.data(),0,output.size()*sizeof(float));
    for(uint32_t f=0;f<input.frames;++f){
        std::memcpy(motion+uint64_t(f)*585,output.data()+uint64_t(f)*588,585*sizeof(float));
        std::memcpy(camera+uint64_t(f)*3,output.data()+uint64_t(f)*588+585,3*sizeof(float));
        camera[uint64_t(f)*3]=std::max(camera[uint64_t(f)*3],.25f);
    }
    profile_.download_ns+=elapsed(download_started);
}

void session::infer(const gemx_sequence_view &input,float *motion,float *camera){
    std::lock_guard lock(mutex_);
    require(input.frames>=1 && input.frames<=GEMX_MAX_FRAMES,
            "native denoiser accepts 1..4096 frames");
    require(motion && camera,"inference output buffers are required");
    require(input.keypoints && input.boxes && input.intrinsics && input.body_features &&
            input.camera_angular_velocity,"all sequence arrays are required");
    for(uint32_t start=0;start<input.frames;start+=120){
        const uint32_t count=std::min<uint32_t>(120,input.frames-start);
        gemx_sequence_view window{count,
            input.keypoints+uint64_t(start)*77*3,input.boxes+uint64_t(start)*3,
            input.intrinsics+uint64_t(start)*9,input.body_features+uint64_t(start)*1024,
            input.camera_angular_velocity+uint64_t(start)*6};
        infer_window(window,motion+uint64_t(start)*585,camera+uint64_t(start)*3);
    }
    ++profile_.calls;profile_.frames+=input.frames;
}

gemx_profile session::profile() const{std::lock_guard lock(mutex_);return profile_;}
}
