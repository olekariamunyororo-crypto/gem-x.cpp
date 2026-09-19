#include "session.hpp"
#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace gemx { namespace {
using mat=std::array<float,9>;using vec=std::array<float,3>;using quat=std::array<float,4>;
mat transposed(const mat &a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
mat product(const mat &a,const mat &b){
    mat out{};
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)for(int k=0;k<3;++k)
        out[r*3+c]+=a[r*3+k]*b[k*3+c];
    return out;
}
vec product(const mat &a,const vec &b){return {a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
    a[3]*b[0]+a[4]*b[1]+a[5]*b[2],a[6]*b[0]+a[7]*b[1]+a[8]*b[2]};}
mat rotation(const float *angle){
    float length=std::sqrt(angle[0]*angle[0]+angle[1]*angle[1]+angle[2]*angle[2]);
    float half=.5f*length,factor=length<1e-6f?.5f-length*length/48.f:std::sin(half)/length;
    float w=std::cos(half),x=angle[0]*factor,y=angle[1]*factor,z=angle[2]*factor;
    float s=2.f/(w*w+x*x+y*y+z*z);
    return {1-s*(y*y+z*z),s*(x*y-z*w),s*(x*z+y*w),s*(x*y+z*w),
        1-s*(x*x+z*z),s*(y*z-x*w),s*(x*z-y*w),s*(y*z+x*w),1-s*(x*x+y*y)};
}
mat rest_rotation(const std::vector<float> &transforms,uint32_t joint){
    const float *m=transforms.data()+uint64_t(joint)*16;
    return {m[0],m[1],m[2],m[4],m[5],m[6],m[8],m[9],m[10]};
}
quat quaternion(const mat &m){
    std::array<float,4> qabs={std::sqrt(std::max(0.f,1+m[0]+m[4]+m[8])),
        std::sqrt(std::max(0.f,1+m[0]-m[4]-m[8])),std::sqrt(std::max(0.f,1-m[0]+m[4]-m[8])),
        std::sqrt(std::max(0.f,1-m[0]-m[4]+m[8]))};
    std::array<std::array<float,4>,4> candidates={{{qabs[0]*qabs[0],m[7]-m[5],m[2]-m[6],m[3]-m[1]},
        {m[7]-m[5],qabs[1]*qabs[1],m[3]+m[1],m[2]+m[6]},
        {m[2]-m[6],m[3]+m[1],qabs[2]*qabs[2],m[5]+m[7]},
        {m[3]-m[1],m[6]+m[2],m[7]+m[5],qabs[3]*qabs[3]}}};
    size_t choice=std::max_element(qabs.begin(),qabs.end())-qabs.begin();
    float divisor=2*std::max(qabs[choice],.1f);auto q=candidates[choice];
    for(float &value:q)value/=divisor;
    if(q[0]<0)for(float &value:q)value=-value;
    float length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    return {q[1]/length,q[2]/length,q[3]/length,q[0]/length};
}
}

skeleton_data session::skeleton(const gemx_motion_view &motion) const{
    std::lock_guard lock(mutex_);
    require(motion.frames>=1 && motion.frames<=GEMX_MAX_FRAMES,"skeleton frames are invalid");
    require(motion.body_pose && motion.identity_coeffs && motion.scale_params &&
            motion.global_orient_world && motion.translation_world,
            "world motion arrays are required for skeleton construction");
    require(soma_local_.size()==78*16 && soma_world_.size()==78*16 &&
            soma_parents_.size()==78 && soma_names_.size()==78,"SOMA rig is unavailable");
    skeleton_data result;result.frames=motion.frames;
    result.positions.resize(uint64_t(motion.frames)*77*3);
    result.local_rotations.resize(uint64_t(motion.frames)*77*4);
    result.local_translations.resize(uint64_t(motion.frames)*77*3);
    result.parents.resize(77);result.names.assign(soma_names_.begin()+1,soma_names_.end());
    for(uint32_t joint=1;joint<78;++joint){
        int32_t parent=soma_parents_[joint];require(parent>=0 && parent<static_cast<int32_t>(joint),
                                                    "SOMA hierarchy is not topological");
        result.parents[joint-1]=parent==0?-1:parent-1;
    }
    std::array<mat,78> orient{};
    for(uint32_t joint=0;joint<78;++joint)
        orient[joint]=rest_rotation(soma_world_,joint);
    std::array<float,45> identity{};std::array<float,69> scales{};
    for(uint32_t frame=0;frame<motion.frames;++frame){
        for(uint32_t i=0;i<45;++i)identity[i]+=motion.identity_coeffs[uint64_t(frame)*45+i];
        for(uint32_t i=0;i<69;++i)scales[i]+=motion.scale_params[uint64_t(frame)*69+i];
    }
    for(float &value:identity)value/=motion.frames;
    for(float &value:scales)value/=motion.frames;
    // Reuse only bit-identical inputs: no shape smoothing, quantization or
    // tolerance-based cache key. A new frame's shape still gets its own fit.
    if(!shape_cached_ || std::memcmp(identity.data(),cached_identity_.data(),sizeof(identity)) ||
       std::memcmp(scales.data(),cached_scales_.data(),sizeof(scales))){
        auto fitted=fit_soma_identity(soma_identity_,identity.data(),scales.data()+1,scales[0]);
        cached_shape_=std::move(fitted);cached_identity_=identity;cached_scales_=scales;shape_cached_=true;
    }
    const auto &fitted=cached_shape_;
    for(uint32_t frame=0;frame<motion.frames;++frame){
        std::array<mat,77> world_rotation{};std::array<vec,77> world_position{};
        for(uint32_t joint=1;joint<78;++joint){
            const float *pose=joint==1?motion.global_orient_world+uint64_t(frame)*3:
                motion.body_pose+(uint64_t(frame)*76+(joint-2))*3;
            mat local_rotation=product(product(transposed(orient[soma_parents_[joint]]),rotation(pose)),
                                       orient[joint]);
            vec translation;
            if(joint==1){
                const float *value=motion.translation_world+uint64_t(frame)*3;
                translation={value[0],value[1],value[2]};
            }else{
                const float *value=fitted.local_offsets.data()+uint64_t(joint)*3;
                translation={value[0],value[1],value[2]};
            }
            const uint64_t output_joint=joint-1;
            std::copy(translation.begin(),translation.end(),result.local_translations.begin()+
                      (uint64_t(frame)*77+output_joint)*3);
            auto q=quaternion(local_rotation);std::copy(q.begin(),q.end(),result.local_rotations.begin()+
                      (uint64_t(frame)*77+output_joint)*4);
            int32_t parent=result.parents[output_joint];
            if(parent<0){
                world_rotation[output_joint]=local_rotation;
                world_position[output_joint]=translation;
            }else{
                world_rotation[output_joint]=product(world_rotation[parent],local_rotation);
                auto offset=product(world_rotation[parent],translation);
                for(int i=0;i<3;++i)world_position[output_joint][i]=world_position[parent][i]+offset[i];
            }
            std::copy(world_position[output_joint].begin(),world_position[output_joint].end(),
                      result.positions.begin()+(uint64_t(frame)*77+output_joint)*3);
        }
    }
    return result;
}

void session::export_skeleton(const gemx_skeleton_view &input,float fps,const std::string &path) const{
    std::lock_guard lock(mutex_);
    require(input.frames>=1 && input.frames<=GEMX_MAX_FRAMES && input.joint_positions &&
            input.local_rotations && input.local_translations && input.parents,
            "complete SOMA skeleton samples are required");
    require(soma_names_.size()==78 && soma_parents_.size()==78,"SOMA topology is unavailable");
    skeleton_data value;value.frames=input.frames;
    value.positions.assign(input.joint_positions,input.joint_positions+uint64_t(input.frames)*77*3);
    value.local_rotations.assign(input.local_rotations,input.local_rotations+uint64_t(input.frames)*77*4);
    value.local_translations.assign(input.local_translations,input.local_translations+uint64_t(input.frames)*77*3);
    value.parents.assign(input.parents,input.parents+77);
    value.names.assign(soma_names_.begin()+1,soma_names_.end());
    for(uint32_t joint=0;joint<77;++joint){
        const int32_t source=soma_parents_[joint+1];
        require(value.parents[joint]==(source==0?-1:source-1),
                "skeleton topology does not match this model");
    }
    write_glb(value,fps,path);
}
}
