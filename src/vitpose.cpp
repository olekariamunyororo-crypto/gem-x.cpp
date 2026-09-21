// Native graph equations follow NVIDIA GEM-X's Apache-2.0 ViTPose export.
// The DINOv3 backbone architecture originates with Meta; see NOTICE.
#include "vitpose.hpp"
#include "internal.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <future>
#include <limits>

namespace gemx { namespace {
constexpr int64_t image_width=192,image_height=256,patch=16,grid_width=12,grid_height=16;
constexpr int64_t embedding=1280,heads=20,head_dim=64,hidden=5120,prefix_tokens=5,tokens=197;
constexpr int64_t joints=77,heat_width=48,heat_height=64;

ggml_tensor *linear(ggml_context *ctx,const model &weights,const std::string &prefix,
                    ggml_tensor *input){
    // Preserve token/crop ordering while presenting all
    // rows to one GEMM. Reshapes are views and do not copy the activations.
    static const bool flatten=[] {
        const char *value=std::getenv("GEMX_VITPOSE_FLATTEN");
        return !value || std::strcmp(value,"1")==0;
    }();
    auto *matrix=input;
    if(flatten && input->ne[2]==2 && input->ne[3]==1 && ggml_is_contiguous(input))
        matrix=ggml_reshape_2d(ctx,input,input->ne[0],input->ne[1]*input->ne[2]*input->ne[3]);
    auto *value=ggml_mul_mat(ctx,weights.tensor(prefix+".weight"),matrix);
    ggml_mul_mat_set_prec(value,GGML_PREC_F32);
    if(matrix!=input)
        value=ggml_reshape_4d(ctx,value,value->ne[0],input->ne[1],input->ne[2],input->ne[3]);
    return ggml_add(ctx,value,weights.tensor(prefix+".bias"));
}

ggml_tensor *norm(ggml_context *ctx,const model &weights,const std::string &prefix,
                  ggml_tensor *input){
    auto *value=ggml_norm(ctx,input,1e-6f);
    value=ggml_mul(ctx,value,weights.tensor(prefix+".weight"));
    return ggml_add(ctx,value,weights.tensor(prefix+".bias"));
}

ggml_tensor *channel_bias(ggml_context *ctx,ggml_tensor *input,ggml_tensor *bias){
    return ggml_add(ctx,input,ggml_reshape_4d(ctx,bias,1,1,input->ne[2],1));
}

ggml_tensor *crop_border(ggml_context *ctx,ggml_tensor *input){
    return ggml_cont(ctx,ggml_view_4d(ctx,input,input->ne[0]-2,input->ne[1]-2,
        input->ne[2],input->ne[3],input->nb[1],input->nb[2],input->nb[3],
        input->nb[0]+input->nb[1]));
}

std::vector<float> prepare_frame(const gemx_rgb_frame &frame){
    require(frame.rgb && frame.width && frame.height && frame.width<32767 && frame.height<32767,
            "invalid ViTPose RGB frame");
    const uint64_t row=uint64_t(frame.width)*3;
    require(frame.row_stride>=row && (frame.height==1 ||
            frame.row_stride<=(UINT64_MAX-row)/(frame.height-1)),"ViTPose RGB span overflow");
    const uint64_t span=frame.row_stride*(frame.height-1)+row;
    require(frame.capacity>=span,"ViTPose RGB buffer is too small");
    const float cx=frame.box[0],cy=frame.box[1],size=frame.box[2];
    require(std::isfinite(cx) && std::isfinite(cy) && std::isfinite(size) && size>0,
            "ViTPose box must be finite with positive size");
    // Upstream builds float32 source anchors, then OpenCV computes the affine
    // matrix in double precision. Preserve those rounded anchors, including
    // the tiny shear possible when the centre and endpoints round differently.
    const double origin_x=float(double(cx)-double(size)*.5);
    const double origin_y=float(double(cy)-double(size)*.5);
    const double right_x=float(double(cx)+double(size)*.5);
    const double scale_x=(right_x-origin_x)/255.0;
    const double shear_x=(double(cx)-origin_x)/127.5-scale_x;
    const double scale_y=(double(cy)-origin_y)/127.5;
    std::vector<float> result(3*image_width*image_height);
    constexpr std::array<float,3> mean{.485f,.456f,.406f},stddev{.229f,.224f,.225f};
    auto sample=[&](int64_t x,int64_t y,int channel){
        if(x<0 || y<0 || x>=frame.width || y>=frame.height)return 0;
        return int(frame.rgb[uint64_t(y)*frame.row_stride+uint64_t(x)*3+channel]);
    };
    for(int64_t y=0;y<image_height;++y)for(int64_t x=0;x<image_width;++x){
        // Match OpenCV INTER_LINEAR's 5-bit fractional-coordinate table. The
        // network sees columns 32..223 of the published 256x256 crop.
        // warpAffine rounds the row origin and column increment separately,
        // before adding the interpolation-table half step. Rounding their sum
        // instead changes crop pixels even with identical image and box inputs.
        const int64_t map_x=(int64_t(std::nearbyint((origin_x+shear_x*y)*1024.0))+
            int64_t(std::nearbyint(scale_x*(x+32)*1024.0))+16)>>5;
        const int64_t map_y=(int64_t(std::nearbyint((origin_y+scale_y*y)*1024.0))+16)>>5;
        const int64_t sx=map_x>>5,sy=map_y>>5;const int fx=int(map_x&31),fy=int(map_y&31);
        for(int c=0;c<3;++c){
            // GEM-X's get_batch/vitpose_preprocess reverses the channels of
            // frames supplied by its RGB video reader. Preserve that released
            // convention here; matching the network alone does not cover it.
            const int channel=2-c;
            const int sum=sample(sx,sy,channel)*(32-fx)*(32-fy)+sample(sx+1,sy,channel)*fx*(32-fy)+
                sample(sx,sy+1,channel)*(32-fx)*fy+sample(sx+1,sy+1,channel)*fx*fy;
            const auto pixel=uint8_t((sum+512)>>10);
            result[(c*image_height+y)*image_width+x]=(float(pixel)/255.f-mean[c])/stddev[c];
        }
    }
    return result;
}

constexpr std::array<std::pair<int,int>,32> flip_pairs{{
    {9,10},{11,39},{12,40},{13,41},{14,42},{15,43},{16,44},{17,45},
    {18,46},{19,47},{20,48},{21,49},{22,50},{23,51},{24,52},{25,53},
    {26,54},{27,55},{28,56},{29,57},{30,58},{31,59},{32,60},{33,61},
    {34,62},{35,63},{36,64},{37,65},{38,66},{67,72},{68,73},{69,74}
}};
int flipped_joint(int joint){
    if(joint==70)return 75;
    if(joint==75)return 70;
    if(joint==71)return 76;
    if(joint==76)return 71;
    for(auto [left,right]:flip_pairs){if(joint==left)return right;if(joint==right)return left;}
    return joint;
}
}

std::vector<float> vitpose_prepare_rgb(const gemx_rgb_frame &frame){return prepare_frame(frame);}

struct vitpose::graph_state {
    ggml_context *context=nullptr;ggml_cgraph *graph=nullptr;ggml_gallocr_t allocator=nullptr;
    ggml_tensor *input=nullptr,*output=nullptr;uint32_t batch=0;
    ~graph_state(){if(allocator)ggml_gallocr_free(allocator);if(context)ggml_free(context);}
};

vitpose::vitpose(const gemx_session_config &config){
    require(config.model_path && config.backend_module && config.backend_name,
            "ViTPose model, backend module and backend name are required");
    require(config.threads>=1 && config.threads<=1024,"ViTPose threads must be in 1..1024");
    backend_=std::make_unique<backend>(config.backend_module,config.backend_name,config.device_index,
        config.threads,config.expected_device_description?config.expected_device_description:"");
    model_=std::make_unique<model>(config.model_path,backend_->buffer_type(),"gemx_vitpose",525);
    require(model_->u32("gemx.vitpose.image_width")==192 &&
        model_->u32("gemx.vitpose.image_height")==256 &&
        model_->u32("gemx.vitpose.embedding_length")==1280 &&
        model_->u32("gemx.vitpose.feed_forward_length")==5120 &&
        model_->u32("gemx.vitpose.block_count")==32 &&
        model_->u32("gemx.vitpose.attention.head_count")==20 &&
        model_->u32("gemx.vitpose.joint_count")==77 &&
        model_->f32("gemx.vitpose.layer_norm_epsilon")==1e-6f,
        "unsupported ViTPose model architecture");
    const char *pack=std::getenv("GEMX_VITPOSE_PACK_FFN");
    if(pack && std::strcmp(pack,"1")==0)model_->pack_vitpose_gate_up(backend_->buffer_type());
}
vitpose::~vitpose()=default;

vitpose::graph_state &vitpose::graph(uint32_t batch){
    if(auto found=graphs_.find(batch);found!=graphs_.end())return *found->second;
    require(batch>=1 && batch<=8,"ViTPose batch must be in 1..8");
    auto state=std::make_unique<graph_state>();state->batch=batch;
    state->context=ggml_init({32u*1024u*1024u,nullptr,true});
    if(!state->context)throw std::bad_alloc();
    auto *ctx=state->context;
    state->input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,image_width,image_height,3,batch);
    ggml_set_input(state->input);ggml_set_name(state->input,"vitpose.input");
    auto *patch_weight=model_->tensor("backbone.patch.weight");
    auto *columns=ggml_im2col(ctx,patch_weight,state->input,patch,patch,0,0,1,1,true,GGML_TYPE_F32);
    auto *x=ggml_mul_mat(ctx,ggml_reshape_2d(ctx,patch_weight,3*patch*patch,embedding),
        ggml_reshape_2d(ctx,columns,3*patch*patch,grid_width*grid_height*batch));
    ggml_mul_mat_set_prec(x,GGML_PREC_F32);
    x=ggml_add(ctx,x,model_->tensor("backbone.patch.bias"));
    x=ggml_reshape_3d(ctx,x,embedding,grid_width*grid_height,batch);
    auto *cls=ggml_repeat_4d(ctx,model_->tensor("backbone.cls_token"),embedding,1,batch,1);
    auto *storage=ggml_repeat_4d(ctx,model_->tensor("backbone.storage_tokens"),embedding,4,batch,1);
    x=ggml_concat(ctx,ggml_concat(ctx,cls,storage,1),x,1);
    auto *angles=model_->tensor("backbone.rope_angles");
    auto *sin=ggml_sin(ctx,angles),*cos=ggml_cos(ctx,angles);
    constexpr float operand_scale=.3535533905932738f;
    const char *flash_env=std::getenv("GEMX_VITPOSE_FLASH_ATTN");
    const bool flash_attention=flash_env && std::strcmp(flash_env,"1")==0;
    for(int block=0;block<32;++block){
        const std::string p="backbone.block."+std::to_string(block);
        auto *normalized=norm(ctx,*model_,p+".norm1",x);
        auto *qkv=linear(ctx,*model_,p+".attn.qkv",normalized);
        auto head=[&](int index){
            auto *value=ggml_view_4d(ctx,qkv,head_dim,heads,tokens,batch,
                head_dim*sizeof(float),3*embedding*sizeof(float),
                3*embedding*tokens*sizeof(float),index*embedding*sizeof(float));
            return ggml_cont(ctx,ggml_permute(ctx,value,0,2,1,3));
        };
        auto *q=head(0),*k=head(1),*v=head(2);
        auto rotate=[&](ggml_tensor *value){
            auto *first=ggml_view_4d(ctx,value,head_dim/2,tokens,heads,batch,
                value->nb[1],value->nb[2],value->nb[3],0);
            auto *second=ggml_view_4d(ctx,value,head_dim/2,tokens,heads,batch,
                value->nb[1],value->nb[2],value->nb[3],head_dim/2*sizeof(float));
            return ggml_add(ctx,ggml_mul(ctx,value,cos),
                ggml_mul(ctx,ggml_concat(ctx,ggml_neg(ctx,second),first,0),sin));
        };
        q=ggml_cont(ctx,rotate(q));k=ggml_cont(ctx,rotate(k));
        ggml_tensor *attended=nullptr;
        if(flash_attention){
            // Preserve upstream's separate Q/K scaling and F32 storage. Flash
            // attention changes the reduction order, so remains opt-in.
            attended=ggml_flash_attn_ext(ctx,ggml_scale(ctx,q,operand_scale),
                ggml_scale(ctx,k,operand_scale),v,nullptr,1.f,0.f,0.f);
            ggml_flash_attn_ext_set_prec(attended,GGML_PREC_F32);
            // FA already returns [head_dim, heads, tokens, batch].
            attended=ggml_reshape_3d(ctx,attended,embedding,tokens,batch);
        }else{
            auto *scores=ggml_mul_mat(ctx,ggml_scale(ctx,k,operand_scale),ggml_scale(ctx,q,operand_scale));
            ggml_mul_mat_set_prec(scores,GGML_PREC_F32);scores=ggml_soft_max(ctx,scores);
            attended=ggml_mul_mat(ctx,ggml_cont(ctx,ggml_transpose(ctx,v)),scores);
            ggml_mul_mat_set_prec(attended,GGML_PREC_F32);
            attended=ggml_reshape_3d(ctx,ggml_cont(ctx,ggml_permute(ctx,attended,0,2,1,3)),
                embedding,tokens,batch);
        }
        attended=linear(ctx,*model_,p+".attn.proj",attended);
        x=ggml_add(ctx,x,ggml_mul(ctx,attended,model_->tensor(p+".ls1.gamma")));
        normalized=norm(ctx,*model_,p+".norm2",x);
        ggml_tensor *mlp=nullptr;
        if(model_->contains(p+".mlp.w12.weight")){
            auto *gate_up=linear(ctx,*model_,p+".mlp.w12",normalized);
            mlp=ggml_swiglu(ctx,gate_up);
        }else{
            auto *w1=linear(ctx,*model_,p+".mlp.w1",normalized);
            auto *w2=linear(ctx,*model_,p+".mlp.w2",normalized);
            const char *fused_glu=std::getenv("GEMX_VITPOSE_SWIGLU");
            mlp=(!fused_glu || std::strcmp(fused_glu,"1")==0) ?
                ggml_swiglu_split(ctx,w1,w2) : ggml_mul(ctx,ggml_silu(ctx,w1),w2);
        }
        mlp=linear(ctx,*model_,p+".mlp.w3",mlp);
        x=ggml_add(ctx,x,ggml_mul(ctx,mlp,model_->tensor(p+".ls2.gamma")));
    }
    x=norm(ctx,*model_,"backbone.norm",x);
    x=ggml_view_4d(ctx,x,embedding,grid_width,grid_height,batch,
        embedding*sizeof(float),embedding*grid_width*sizeof(float),
        embedding*tokens*sizeof(float),prefix_tokens*embedding*sizeof(float));
    x=ggml_cont(ctx,ggml_permute(ctx,x,2,0,1,3));
    x=ggml_conv_transpose_2d_p0(ctx,model_->tensor("head.deconv.0.weight"),x,2);
    x=ggml_relu(ctx,channel_bias(ctx,crop_border(ctx,x),model_->tensor("head.deconv.0.bias")));
    x=ggml_conv_transpose_2d_p0(ctx,model_->tensor("head.deconv.1.weight"),x,2);
    x=ggml_relu(ctx,channel_bias(ctx,crop_border(ctx,x),model_->tensor("head.deconv.1.bias")));
    x=ggml_cont(ctx,ggml_permute(ctx,x,1,2,0,3));
    auto *heat=ggml_mul_mat(ctx,ggml_reshape_2d(ctx,model_->tensor("head.final.weight"),256,joints),
        ggml_reshape_2d(ctx,x,256,heat_width*heat_height*batch));
    ggml_mul_mat_set_prec(heat,GGML_PREC_F32);
    heat=ggml_add(ctx,heat,model_->tensor("head.final.bias"));
    heat=ggml_reshape_4d(ctx,heat,joints,heat_width,heat_height,batch);
    state->output=ggml_cont(ctx,ggml_permute(ctx,heat,2,0,1,3));
    ggml_set_output(state->output);ggml_set_name(state->output,"vitpose.heatmaps");
    state->graph=ggml_new_graph_custom(ctx,8192,false);ggml_build_forward_expand(state->graph,state->output);
    state->allocator=ggml_gallocr_new(backend_->buffer_type());
    if(!state->allocator || !ggml_gallocr_alloc_graph(state->allocator,state->graph))throw std::bad_alloc();
    for(int i=0;i<ggml_graph_n_nodes(state->graph);++i)
        require(ggml_backend_supports_op(backend_->handle(),ggml_graph_node(state->graph,i)),
                "backend cannot execute ViTPose graph operation");
    auto [inserted,_]=graphs_.emplace(batch,std::move(state));return *inserted->second;
}

void vitpose::infer_normalized(const float *images,uint32_t batch,float *heatmaps){
    require(images && heatmaps && batch>=1 && batch<=8,"invalid normalized ViTPose inference");
    const uint64_t input_count=uint64_t(batch)*3*image_height*image_width;
    require(std::all_of(images,images+input_count,[](float value){return std::isfinite(value);}),
            "non-finite normalized ViTPose image");
    std::lock_guard lock(mutex_);auto &state=graph(batch);
    ggml_backend_tensor_set(state.input,images,0,input_count*sizeof(float));
    backend_->compute(state.graph);
    ggml_backend_tensor_get(state.output,heatmaps,0,uint64_t(batch)*joints*heat_width*heat_height*sizeof(float));
    require(std::all_of(heatmaps,heatmaps+uint64_t(batch)*joints*heat_width*heat_height,
        [](float value){return std::isfinite(value);}),"non-finite ViTPose heatmap");
}

void vitpose::infer_rgb(const gemx_rgb_frame *frames,uint32_t count,float *keypoints){
    require(frames && keypoints && count>=1 && count<=4,"ViTPose RGB frame count must be in 1..4");
    std::vector<std::future<std::vector<float>>> work;work.reserve(count);
    // A single live crop has no parallel preparation to overlap. Avoid spawning
    // and joining a thread for every camera frame; keep offline batch parallelism.
    for(uint32_t i=0;count>1 && i<count;++i)work.push_back(std::async(std::launch::async,[&,i]{return prepare_frame(frames[i]);}));
    const uint64_t image_elements=3*image_height*image_width;
    std::vector<float> images(uint64_t(count)*2*image_elements);
    for(uint32_t frame=0;frame<count;++frame){
        auto image=count==1?prepare_frame(frames[frame]):work[frame].get();std::copy(image.begin(),image.end(),images.begin()+uint64_t(frame)*2*image_elements);
        auto *flipped=images.data()+(uint64_t(frame)*2+1)*image_elements;
        for(int c=0;c<3;++c)for(int y=0;y<image_height;++y)for(int x=0;x<image_width;++x)
            flipped[(c*image_height+y)*image_width+x]=image[(c*image_height+y)*image_width+(image_width-1-x)];
    }
    std::vector<float> heatmaps(uint64_t(count)*2*joints*heat_width*heat_height);
    infer_normalized(images.data(),count*2,heatmaps.data());
    const uint64_t plane=heat_width*heat_height,frame_stride=joints*plane;
    for(uint32_t frame=0;frame<count;++frame)for(int joint=0;joint<joints;++joint){
        const float *plain=heatmaps.data()+uint64_t(frame*2)*frame_stride+joint*plane;
        const float *flipped=heatmaps.data()+uint64_t(frame*2+1)*frame_stride+flipped_joint(joint)*plane;
        float maximum=-std::numeric_limits<float>::infinity();int best=0;
        auto value=[&](int x,int y){return .5f*(plain[y*heat_width+x]+flipped[y*heat_width+(heat_width-1-x)]);};
        for(uint64_t i=0;i<plane;++i){float candidate=.5f*(plain[i]+flipped[(i/heat_width)*heat_width+(heat_width-1-i%heat_width)]);
            if(candidate>maximum){maximum=candidate;best=i;}}
        const int ix=best%heat_width,iy=best/heat_width;float x=float(ix),y=float(iy);
        auto quarter=[](float difference){return difference>0?.25f:(difference<0?-.25f:0.f);};
        if(ix>1 && ix<heat_width-1)x+=quarter(value(ix+1,iy)-value(ix-1,iy));
        if(iy>1 && iy<heat_height-1)y+=quarter(value(ix,iy+1)-value(ix,iy-1));
        const float size=frames[frame].box[2],cx=frames[frame].box[0],cy=frames[frame].box[1];
        auto *output=keypoints+(uint64_t(frame)*joints+joint)*3;
        output[0]=x/float(heat_width)*(size*.75f)+cx-size*.375f;
        output[1]=y/float(heat_height)*size+cy-size*.5f;output[2]=maximum;
    }
}
}
