#include "session.hpp"
#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace gemx { namespace {
using matrix=std::array<float,9>;
using vector=std::array<float,3>;

float norm(vector value){return std::sqrt(value[0]*value[0]+value[1]*value[1]+value[2]*value[2]);}
vector normalize_vector(vector value,float epsilon=1e-12f){
    float divisor=std::max(norm(value),epsilon);
    for(float &item:value)item/=divisor;
    return value;
}
matrix transpose(const matrix &a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
matrix multiply(const matrix &a,const matrix &b){
    matrix result{};
    for(int row=0;row<3;++row)for(int column=0;column<3;++column)
        for(int k=0;k<3;++k)result[row*3+column]+=a[row*3+k]*b[k*3+column];
    return result;
}
vector multiply(const matrix &a,const vector &b){
    return {a[0]*b[0]+a[1]*b[1]+a[2]*b[2],a[3]*b[0]+a[4]*b[1]+a[5]*b[2],
            a[6]*b[0]+a[7]*b[1]+a[8]*b[2]};
}
matrix rotation6(const float *value){
    vector first=normalize_vector({value[0],value[1],value[2]});
    vector second={value[3],value[4],value[5]};
    float dot=first[0]*second[0]+first[1]*second[1]+first[2]*second[2];
    for(int i=0;i<3;++i)second[i]-=dot*first[i];
    second=normalize_vector(second);
    vector third={first[1]*second[2]-first[2]*second[1],
                  first[2]*second[0]-first[0]*second[2],
                  first[0]*second[1]-first[1]*second[0]};
    return {first[0],first[1],first[2],second[0],second[1],second[2],third[0],third[1],third[2]};
}
matrix axis_angle_matrix(const vector &angle){
    float magnitude=norm(angle),half=.5f*magnitude;
    float factor=std::abs(magnitude)<1e-6f?.5f-magnitude*magnitude/48.f:std::sin(half)/magnitude;
    float w=std::cos(half),x=angle[0]*factor,y=angle[1]*factor,z=angle[2]*factor;
    float two_s=2.f/(w*w+x*x+y*y+z*z);
    return {1.f-two_s*(y*y+z*z),two_s*(x*y-z*w),two_s*(x*z+y*w),
            two_s*(x*y+z*w),1.f-two_s*(x*x+z*z),two_s*(y*z-x*w),
            two_s*(x*z-y*w),two_s*(y*z+x*w),1.f-two_s*(x*x+y*y)};
}
vector matrix_axis_angle(const matrix &m){
    std::array<float,4> qabs={
        std::sqrt(std::max(0.f,1.f+m[0]+m[4]+m[8])),
        std::sqrt(std::max(0.f,1.f+m[0]-m[4]-m[8])),
        std::sqrt(std::max(0.f,1.f-m[0]+m[4]-m[8])),
        std::sqrt(std::max(0.f,1.f-m[0]-m[4]+m[8]))};
    std::array<std::array<float,4>,4> candidates={{
        {qabs[0]*qabs[0],m[7]-m[5],m[2]-m[6],m[3]-m[1]},
        {m[7]-m[5],qabs[1]*qabs[1],m[3]+m[1],m[2]+m[6]},
        {m[2]-m[6],m[3]+m[1],qabs[2]*qabs[2],m[5]+m[7]},
        {m[3]-m[1],m[6]+m[2],m[7]+m[5],qabs[3]*qabs[3]}}};
    auto chosen=static_cast<size_t>(std::max_element(qabs.begin(),qabs.end())-qabs.begin());
    float denominator=2.f*std::max(qabs[chosen],.1f);
    auto q=candidates[chosen];for(float &value:q)value/=denominator;
    if(q[0]<0)for(float &value:q)value=-value;
    float xyz=std::sqrt(q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    float half=std::atan2(xyz,q[0]);
    float divisor=std::abs(half)<1e-8f?.5f:.5f*std::sin(half)/half;
    return {q[1]/divisor,q[2]/divisor,q[3]/divisor};
}
void store(const vector &value,float *target){std::copy(value.begin(),value.end(),target);}

std::vector<vector> gaussian_smooth(const std::vector<vector> &values){
    constexpr int radius=12;constexpr double sigma=3.;
    std::array<float,2*radius+1> kernel{};double total=0;
    for(int i=-radius;i<=radius;++i)total+=std::exp(-.5*double(i*i)/(sigma*sigma));
    for(int i=-radius;i<=radius;++i)
        kernel[i+radius]=static_cast<float>(std::exp(-.5*double(i*i)/(sigma*sigma))/total);
    std::vector<vector> result(values.size());
    for(size_t frame=0;frame<values.size();++frame)for(int component=0;component<3;++component){
        float sum=0;
        for(int offset=-radius;offset<=radius;++offset){
            auto index=std::clamp<int64_t>(static_cast<int64_t>(frame)+offset,0,
                                           static_cast<int64_t>(values.size())-1);
            sum+=values[index][component]*kernel[offset+radius];
        }
        result[frame][component]=sum;
    }
    return result;
}
}

void session::decode(const gemx_sequence_view &input,const float *normalized,const float *camera,
                     const gemx_motion_view &output){
    std::lock_guard lock(mutex_);
    require(output.frames==input.frames,"decoded output frame count does not match input");
    require(output.body_pose && output.identity_coeffs && output.scale_params &&
            output.global_orient_camera && output.translation_camera &&
            output.global_orient_world && output.translation_world,
            "all decoded motion output arrays are required");
    require(motion_mean_.size()==585 && motion_std_.size()==585,"motion statistics are unavailable");
    std::vector<float> decoded(uint64_t(input.frames)*585);
    for(uint32_t frame=0;frame<input.frames;++frame)for(uint32_t i=0;i<585;++i){
        float value=normalized[uint64_t(frame)*585+i];require(finite(value),"finite motion required");
        decoded[uint64_t(frame)*585+i]=value*motion_std_[i]+motion_mean_[i];
    }
    std::vector<vector> orient_camera(input.frames),orient_gravity(input.frames);
    std::vector<matrix> rotation_camera(input.frames),rotation_gravity(input.frames);
    for(uint32_t frame=0;frame<input.frames;++frame){
        const float *source=decoded.data()+uint64_t(frame)*585;
        for(uint32_t joint=0;joint<76;++joint)
            store(matrix_axis_angle(rotation6(source+joint*6)),output.body_pose+(uint64_t(frame)*76+joint)*3);
        std::copy_n(source+456,45,output.identity_coeffs+uint64_t(frame)*45);
        std::copy_n(source+501,69,output.scale_params+uint64_t(frame)*69);
        output.scale_params[uint64_t(frame)*69]=std::clamp(output.scale_params[uint64_t(frame)*69],.7f,1.f);
        rotation_camera[frame]=rotation6(source+570);orient_camera[frame]=matrix_axis_angle(rotation_camera[frame]);
        rotation_gravity[frame]=rotation6(source+576);orient_gravity[frame]=matrix_axis_angle(rotation_gravity[frame]);
        store(orient_camera[frame],output.global_orient_camera+uint64_t(frame)*3);
        float s=camera[uint64_t(frame)*3],tx=camera[uint64_t(frame)*3+1],ty=camera[uint64_t(frame)*3+2];
        const float *box=input.boxes+uint64_t(frame)*3,*K=input.intrinsics+uint64_t(frame)*9;
        float sb=s*box[2]+1e-9f;
        output.translation_camera[uint64_t(frame)*3]=tx+2.f*(box[0]-K[2])/sb;
        output.translation_camera[uint64_t(frame)*3+1]=ty+2.f*(box[1]-K[5])/sb;
        output.translation_camera[uint64_t(frame)*3+2]=2.f*K[0]/sb;
    }

    std::vector<vector> yaw_delta(input.frames);
    for(uint32_t frame=0;frame<input.frames;++frame){
        matrix camera_delta=rotation6(input.camera_angular_velocity+uint64_t(frame)*6);
        if(norm(matrix_axis_angle(camera_delta))<1e-5f)camera_delta={1,0,0,0,1,0,0,0,1};
        matrix camera_to_gravity=multiply(rotation_gravity[frame],transpose(rotation_camera[frame]));
        matrix next_to_gravity=multiply(camera_to_gravity,transpose(camera_delta));
        vector first=normalize_vector({camera_to_gravity[2],0,camera_to_gravity[8]});
        vector second=normalize_vector({next_to_gravity[2],0,next_to_gravity[8]});
        vector axis=normalize_vector({second[1]*first[2]-second[2]*first[1],
                                second[2]*first[0]-second[0]*first[2],
                                second[0]*first[1]-second[1]*first[0]});
        float angle=std::acos(std::clamp(first[0]*second[0]+first[1]*second[1]+first[2]*second[2],-1.f,1.f));
        yaw_delta[frame]={axis[0]*angle,axis[1]*angle,axis[2]*angle};
    }
    yaw_delta=gaussian_smooth(yaw_delta);
    matrix cumulative={1,0,0,0,1,0,0,0,1};
    std::vector<matrix> world_rotation(input.frames);
    for(uint32_t frame=0;frame<input.frames;++frame){
        if(frame)cumulative=multiply(cumulative,transpose(axis_angle_matrix(yaw_delta[frame])));
        if(norm(matrix_axis_angle(cumulative))<1e-5f)cumulative={1,0,0,0,1,0,0,0,1};
        world_rotation[frame]=multiply(cumulative,rotation_gravity[frame]);
        store(matrix_axis_angle(world_rotation[frame]),output.global_orient_world+uint64_t(frame)*3);
    }
    vector translation={0,0,0};store(translation,output.translation_world);
    for(uint32_t frame=1;frame<input.frames;++frame){
        const float *velocity=decoded.data()+uint64_t(frame-1)*585+582;
        vector local_velocity={velocity[0],velocity[1],velocity[2]};
        vector delta=multiply(world_rotation[frame-1],local_velocity);
        for(int i=0;i<3;++i)translation[i]+=delta[i];
        store(translation,output.translation_world+uint64_t(frame)*3);
    }
}

void session::decode_velocity(const float *motion,float velocity[3]) const{
    std::lock_guard lock(mutex_);
    require(motion && velocity,"motion and velocity outputs are required");
    require(motion_mean_.size()==585 && motion_std_.size()==585,"motion statistics are unavailable");
    for(uint32_t i=0;i<3;++i){
        const uint32_t index=582+i;
        velocity[i]=motion[index]*motion_std_[index]+motion_mean_[index];
        require(finite(velocity[i]),"finite local velocity required");
    }
}
}
