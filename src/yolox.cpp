// Native graph and preprocessing follow the YOLOX-X HumanArt export selected
// by NVIDIA GEM-X. YOLOX is Copyright (c) 2021 Megvii, Apache-2.0.
#include "yolox.hpp"
#include "internal.hpp"
#include "ggml-alloc.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <limits>
#include <numeric>
#include <string_view>

namespace gemx { namespace {
constexpr int image_size=640,focus_size=320,focus_channels=12;

int integer(std::string_view text){
    int value=0;auto result=std::from_chars(text.data(),text.data()+text.size(),value);
    require(result.ec==std::errc{}&&result.ptr==text.data()+text.size(),"invalid YOLOX graph integer");
    return value;
}
std::vector<std::string_view> split(std::string_view value,char delimiter){
    std::vector<std::string_view> result;
    for(size_t start=0;;){
        const size_t end=value.find(delimiter,start);
        result.push_back(value.substr(start,end==std::string_view::npos?value.size()-start:end-start));
        if(end==std::string_view::npos)break;
        start=end+1;
    }
    return result;
}
ggml_tensor *bias(ggml_context *ctx,ggml_tensor *input,ggml_tensor *value){
    return ggml_add(ctx,input,ggml_reshape_4d(ctx,value,1,1,input->ne[2],1));
}
float sigmoid(float x){return x>=0.f?1.f/(1.f+std::exp(-x)):std::exp(x)/(1.f+std::exp(x));}
float overlap(const gemx_detection &a,const gemx_detection &b){
    const float x0=std::max(a.box[0],b.box[0]),y0=std::max(a.box[1],b.box[1]);
    const float x1=std::min(a.box[2],b.box[2]),y1=std::min(a.box[3],b.box[3]);
    const float intersection=std::max(0.f,x1-x0)*std::max(0.f,y1-y0);
    const float aa=std::max(0.f,a.box[2]-a.box[0])*std::max(0.f,a.box[3]-a.box[1]);
    const float ab=std::max(0.f,b.box[2]-b.box[0])*std::max(0.f,b.box[3]-b.box[1]);
    return intersection/(aa+ab-intersection+1e-7f);
}
}

std::vector<float> yolox_prepare_rgb(const gemx_rgb_frame &frame,float &ratio){
    require(frame.rgb&&frame.width&&frame.height&&frame.width<32767&&frame.height<32767,
            "invalid YOLOX RGB frame");
    const uint64_t row=uint64_t(frame.width)*3;
    require(frame.row_stride>=row&&(frame.height==1||frame.row_stride<=(UINT64_MAX-row)/(frame.height-1)),
            "YOLOX RGB span overflow");
    require(frame.capacity>=frame.row_stride*(frame.height-1)+row,"YOLOX RGB buffer is too small");
    const double resize_ratio=std::min(double(image_size)/frame.height,double(image_size)/frame.width);
    ratio=float(resize_ratio);
    const int resized_width=std::max(1,int(frame.width*resize_ratio));
    const int resized_height=std::max(1,int(frame.height*resize_ratio));
    std::vector<float> result(uint64_t(focus_channels)*focus_size*focus_size);
    // Coefficients depend only on destination coordinates, not on channel or
    // Focus quadrant. Compute them once per axis, preserving OpenCV rounding.
    struct axis_sample {int first,second,a0,a1;};
    std::array<axis_sample,image_size> xs{},ys{};
    for(int dx=0;dx<resized_width;++dx){
        float fx=float((double(dx)+.5)*frame.width/resized_width-.5);
        int sx=int(std::floor(fx));fx-=sx;
        if(sx<0){sx=0;fx=0;}else if(sx>=int(frame.width)-1){sx=int(frame.width)-1;fx=0;}
        xs[dx]={sx,std::min(sx+1,int(frame.width)-1),
            int(std::nearbyint((1.f-fx)*2048)),int(std::nearbyint(fx*2048))};
    }
    for(int dy=0;dy<resized_height;++dy){
        float fy=float((double(dy)+.5)*frame.height/resized_height-.5);
        int sy=int(std::floor(fy));fy-=sy;
        ys[dy]={std::clamp(sy,0,int(frame.height)-1),std::clamp(sy+1,0,int(frame.height)-1),
            int(std::nearbyint((1.f-fy)*2048)),int(std::nearbyint(fy*2048))};
    }
    auto pixel=[&](int dx,int dy,int bgr){
        if(dx>=resized_width||dy>=resized_height)return uint8_t(114);
        const auto [sx,sx1,ax0,ax1]=xs[dx];
        const auto [sy0,sy1,ay0,ay1]=ys[dy];
        const int channel=2-bgr;
        auto sample=[&](int x,int y){return int(frame.rgb[uint64_t(y)*frame.row_stride+uint64_t(x)*3+channel]);};
        const int row0=sample(sx,sy0)*ax0+sample(sx1,sy0)*ax1;
        const int row1=sample(sx,sy1)*ax0+sample(sx1,sy1)*ax1;
        return uint8_t((((ay0*(row0>>4))>>16)+((ay1*(row1>>4))>>16)+2)>>2);
    };
    // Write each Focus channel contiguously. This also keeps the row's
    // vertical coefficients invariant across the inner pixel loop.
    for(int xo=0;xo<2;++xo)for(int yo=0;yo<2;++yo)for(int c=0;c<3;++c){
        const int channel=(xo*2+yo)*3+c;
        for(int y=0;y<focus_size;++y)for(int x=0;x<focus_size;++x)
            result[(uint64_t(channel)*focus_size+y)*focus_size+x]=pixel(x*2+xo,y*2+yo,c);
    }
    return result;
}

struct yolox::graph_state {
    ggml_context *context=nullptr;ggml_cgraph *graph=nullptr;ggml_gallocr_t allocator=nullptr;
    ggml_tensor *input=nullptr;std::array<ggml_tensor *,9> output{};
    ~graph_state(){if(allocator)ggml_gallocr_free(allocator);if(context)ggml_free(context);}
};

yolox::yolox(const gemx_session_config &config){
    require(config.model_path&&config.backend_module&&config.backend_name,
            "YOLOX model, backend module and backend name are required");
    require(config.threads>=1&&config.threads<=1024,"YOLOX threads must be in 1..1024");
    backend_=std::make_unique<backend>(config.backend_module,config.backend_name,config.device_index,
        config.threads,config.expected_device_description?config.expected_device_description:"");
    model_=std::make_unique<model>(config.model_path,backend_->buffer_type(),"gemx_yolox",310);
    require(model_->u32("gemx.yolox.image_width")==640&&model_->u32("gemx.yolox.image_height")==640&&
            model_->u32("gemx.yolox.convolution_count")==155&&
            model_->u32("gemx.yolox.graph_node_count")==493&&
            model_->string("gemx.yolox.onnx_sha256")==
                "8e9ea96a176bd48501eaaa77216e49ee30794d2f8ba80c7b9862beca4ea972da",
            "unsupported YOLOX model architecture");
    auto state=std::make_unique<graph_state>();
    state->context=ggml_init({64u*1024u*1024u,nullptr,true});if(!state->context)throw std::bad_alloc();
    auto *ctx=state->context;
    state->input=ggml_new_tensor_4d(ctx,GGML_TYPE_F32,focus_size,focus_size,focus_channels,1);
    ggml_set_input(state->input);ggml_set_name(state->input,"yolox.focus");
    std::vector<ggml_tensor *> nodes;nodes.reserve(493);uint32_t convolution=0;
    auto reference=[&](int index){
        require(index>=-1&&index<int(nodes.size()),"YOLOX graph reference is out of range");
        return index<0?state->input:nodes[size_t(index)];
    };
    const std::string recipe=model_->string("gemx.yolox.graph");
    for(auto line:split(recipe,';')){
        auto field=split(line,',');require(!field.empty()&&field[0].size()==1,"invalid YOLOX graph operation");
        ggml_tensor *value=nullptr;
        switch(field[0][0]){
        case 'C': {
            require(field.size()==3&&convolution<155,"invalid YOLOX convolution recipe");
            auto *input=reference(integer(field[1]));const int stride=integer(field[2]);
            require(stride==1||stride==2,"invalid YOLOX convolution stride");
            char name[32];std::snprintf(name,sizeof(name),"conv.%03u.weight",convolution);
            auto *weight=model_->tensor(name);std::snprintf(name,sizeof(name),"conv.%03u.bias",convolution++);
            auto *offset=model_->tensor(name);
            if(!(weight->type==GGML_TYPE_F32&&
                    (weight->ne[0]==1||weight->ne[0]==3)&&weight->ne[0]==weight->ne[1]&&
                    weight->ne[2]==input->ne[2]&&offset->ne[0]==weight->ne[3]))
                throw std::invalid_argument("invalid YOLOX convolution weights at " +
                    std::to_string(convolution-1)+" (input channels "+std::to_string(input->ne[2])+")");
            // ggml_conv_2d unconditionally rounds its im2col buffer to F16
            // for F32 weights. Backend precision switches cannot undo that.
            // Preserve F32 observations so strict mode is actually F32.
            auto *columns=ggml_im2col(ctx,weight,input,stride,stride,int(weight->ne[0]/2),int(weight->ne[1]/2),1,1,true,GGML_TYPE_F32);
            value=ggml_mul_mat(ctx,
                ggml_reshape_2d(ctx,columns,columns->ne[0],columns->ne[1]*columns->ne[2]*columns->ne[3]),
                ggml_reshape_2d(ctx,weight,weight->ne[0]*weight->ne[1]*weight->ne[2],weight->ne[3]));
            ggml_mul_mat_set_prec(value,GGML_PREC_F32);
            value=ggml_reshape_4d(ctx,value,columns->ne[1],columns->ne[2],columns->ne[3],weight->ne[3]);
            value=ggml_cont(ctx,ggml_permute(ctx,value,0,1,3,2));
            value=bias(ctx,value,offset);break;
        }
        case 'S': require(field.size()==2,"invalid YOLOX sigmoid recipe");value=ggml_sigmoid(ctx,reference(integer(field[1])));break;
        case 'M': require(field.size()==3,"invalid YOLOX multiply recipe");value=ggml_mul(ctx,reference(integer(field[1])),reference(integer(field[2])));break;
        case 'A': require(field.size()==3,"invalid YOLOX add recipe");value=ggml_add(ctx,reference(integer(field[1])),reference(integer(field[2])));break;
        case 'T': {
            require(field.size()==3||field.size()==5,"invalid YOLOX concat recipe");
            value=reference(integer(field[1]));
            for(size_t i=2;i<field.size();++i)value=ggml_concat(ctx,value,reference(integer(field[i])),2);
            break;
        }
        case 'P': {
            require(field.size()==3,"invalid YOLOX pool recipe");const int kernel=integer(field[2]);
            require(kernel==5||kernel==9||kernel==13,"invalid YOLOX pool kernel");
            value=ggml_pool_2d(ctx,reference(integer(field[1])),GGML_OP_POOL_MAX,kernel,kernel,1,1,
                               float(kernel/2),float(kernel/2));break;
        }
        case 'U': require(field.size()==2,"invalid YOLOX upscale recipe");value=ggml_upscale(ctx,reference(integer(field[1])),2,GGML_SCALE_MODE_NEAREST);break;
        default: throw std::invalid_argument("unsupported YOLOX graph operation");
        }
        require(value,"failed to build YOLOX operation");nodes.push_back(value);
    }
    require(nodes.size()==493&&convolution==155,"incomplete YOLOX graph recipe");
    const std::string raw_outputs=model_->string("gemx.yolox.raw_outputs");
    auto output_fields=split(raw_outputs,',');
    require(output_fields.size()==9,"invalid YOLOX output recipe");
    for(size_t i=0;i<9;++i){
        auto *value=reference(integer(output_fields[i]));
        if(i%3==0){
            require(value->ne[2]==80,"invalid YOLOX class head");
            value=ggml_view_4d(ctx,value,value->ne[0],value->ne[1],1,1,
                value->nb[1],value->nb[2],value->nb[3],0);
        }else require(value->ne[2]==(i%3==1?4:1),"invalid YOLOX raw output");
        state->output[i]=value;ggml_set_output(value);
    }
    state->graph=ggml_new_graph_custom(ctx,4096,false);
    for(auto *output:state->output)ggml_build_forward_expand(state->graph,output);
    state->allocator=ggml_gallocr_new(backend_->buffer_type());
    if(!state->allocator||!ggml_gallocr_alloc_graph(state->allocator,state->graph))throw std::bad_alloc();
    for(int i=0;i<ggml_graph_n_nodes(state->graph);++i)
        require(ggml_backend_supports_op(backend_->handle(),ggml_graph_node(state->graph,i)),
                "backend cannot execute YOLOX graph operation");
    graph_=std::move(state);
}
yolox::~yolox()=default;

std::vector<gemx_detection> yolox::detect(const gemx_rgb_frame &frame,float threshold,float nms_threshold){
    float ratio=0;auto input=yolox_prepare_rgb(frame,ratio);std::lock_guard lock(mutex_);
    ggml_backend_tensor_set(graph_->input,input.data(),0,input.size()*sizeof(float));
    backend_->compute(graph_->graph);
    std::vector<gemx_detection> candidates;
    for(size_t scale=0;scale<3;++scale){
        const int stride=8<<scale,width=image_size/stride,height=width;
        std::vector<float> cls(uint64_t(width)*height),reg(uint64_t(4)*width*height),object(uint64_t(width)*height);
        ggml_backend_tensor_get(graph_->output[scale*3],cls.data(),0,cls.size()*sizeof(float));
        ggml_backend_tensor_get(graph_->output[scale*3+1],reg.data(),0,reg.size()*sizeof(float));
        ggml_backend_tensor_get(graph_->output[scale*3+2],object.data(),0,object.size()*sizeof(float));
        const uint64_t plane=uint64_t(width)*height;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x){
            const uint64_t index=uint64_t(y)*width+x;const float score=sigmoid(cls[index])*sigmoid(object[index]);
            if(score<threshold)continue;
            const float cx=(reg[index]+x)*stride/ratio,cy=(reg[plane+index]+y)*stride/ratio;
            const float w=std::exp(std::clamp(reg[2*plane+index],-20.f,20.f))*stride/ratio;
            const float h=std::exp(std::clamp(reg[3*plane+index],-20.f,20.f))*stride/ratio;
            gemx_detection value{{cx-w*.5f,cy-h*.5f,cx+w*.5f,cy+h*.5f},score};
            if(std::all_of(value.box,value.box+4,[](float v){return std::isfinite(v);}))candidates.push_back(value);
        }
    }
    std::stable_sort(candidates.begin(),candidates.end(),[](const auto &a,const auto &b){return a.score>b.score;});
    std::vector<gemx_detection> result;result.reserve(std::min<size_t>(100,candidates.size()));
    for(const auto &candidate:candidates){
        if(std::none_of(result.begin(),result.end(),[&](const auto &kept){return overlap(candidate,kept)>nms_threshold;})){
            result.push_back(candidate);if(result.size()==100)break;
        }
    }
    return result;
}
}
