#include "gemx.h"
#include "internal.hpp"
#include "session.hpp"
#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

struct gemx_session {std::unique_ptr<gemx::session> implementation;};

gemx_status gemx_session_create(const gemx_session_config *config,gemx_session **output,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(output,"session output is required");
        *output=nullptr;
        gemx::require(config,"session config is required");
        auto result=std::make_unique<gemx_session>();
        result->implementation=std::make_unique<gemx::session>(*config);
        *output=result.release();
    });
}

void gemx_session_destroy(gemx_session *session){delete session;}

gemx_status gemx_session_create_live(const gemx_session_config *config,gemx_session **output,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(output,"session output is required");
        *output=nullptr;
        gemx::require(config,"session config is required");
        auto result=std::make_unique<gemx_session>();
        result->implementation=std::make_unique<gemx::session>(*config,true);
        *output=result.release();
    });
}

gemx_status gemx_infer(gemx_session *session,const gemx_sequence_view *input,
    float *pred_x,uint64_t pred_x_count,float *pred_camera,uint64_t pred_camera_count,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && input && pred_x && pred_camera,
                      "session, input and output buffers are required");
        gemx::require(pred_x_count==uint64_t(input->frames)*585 &&
                      pred_camera_count==uint64_t(input->frames)*3,
                      "inference output counts do not match frames");
        session->implementation->infer(*input,pred_x,pred_camera);
    });
}

gemx_status gemx_session_get_profile(const gemx_session *session,gemx_profile *profile,
    char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && profile,"session and profile are required");
        *profile=session->implementation->profile();
    });
}

gemx_status gemx_infer_contacts(gemx_session *session,const gemx_sequence_view *input,
    float *motion,uint64_t motion_count,float *camera,uint64_t camera_count,
    float *contacts,uint64_t contact_count,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && input && motion && camera && contacts,
                      "session, input and all output arrays required");
        gemx::require(motion_count==uint64_t(input->frames)*585 && camera_count==uint64_t(input->frames)*3 &&
                      contact_count==uint64_t(input->frames)*6,"contact inference output counts do not match frames");
        session->implementation->infer(*input,motion,camera,contacts);
    });
}

gemx_status gemx_refine_contacts(gemx_session *session,const gemx_motion_view *motion,
    const float *contacts,uint64_t contact_count,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && motion && contacts,"session, motion and contacts required");
        gemx::require(contact_count==uint64_t(motion->frames)*6,"contact count does not match frames");
        session->implementation->refine_contacts(*motion,contacts);
    });
}

gemx_status gemx_infer_motion(gemx_session *session,const gemx_sequence_view *input,
    const gemx_motion_view *output,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && input && output,
                      "session, input and decoded output are required");
        std::vector<float> motion(uint64_t(input->frames)*585),camera(uint64_t(input->frames)*3);
        session->implementation->infer(*input,motion.data(),camera.data());
        session->implementation->decode(*input,motion.data(),camera.data(),*output);
    });
}

gemx_status gemx_decode_predictions(gemx_session *session,const gemx_sequence_view *input,
    const float *pred_x,uint64_t pred_x_count,const float *pred_camera,uint64_t pred_camera_count,
    const gemx_motion_view *output,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && input && pred_x && pred_camera && output,
                      "session, input, raw predictions and decoded output are required");
        gemx::require(pred_x_count==uint64_t(input->frames)*585 &&
                      pred_camera_count==uint64_t(input->frames)*3,
                      "raw prediction counts do not match frames");
        session->implementation->decode(*input,pred_x,pred_camera,*output);
    });
}

const char *gemx_session_device(const gemx_session *session){
    return session && session->implementation?session->implementation->device().c_str():nullptr;
}

gemx_status gemx_build_skeleton(gemx_session *session,const gemx_motion_view *motion,
    const gemx_skeleton_view *skeleton,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && motion && skeleton,
                      "session, motion and skeleton are required");
        gemx::require(skeleton->frames==motion->frames && skeleton->joint_positions &&
                      skeleton->local_rotations,"skeleton output dimensions are invalid");
        auto value=session->implementation->skeleton(*motion);
        std::copy(value.positions.begin(),value.positions.end(),skeleton->joint_positions);
        std::copy(value.local_rotations.begin(),value.local_rotations.end(),skeleton->local_rotations);
        if(skeleton->parents)std::copy(value.parents.begin(),value.parents.end(),skeleton->parents);
        if(skeleton->local_translations)
            std::copy(value.local_translations.begin(),value.local_translations.end(),skeleton->local_translations);
    });
}

gemx_status gemx_export_skeleton_samples_glb(gemx_session *session,const gemx_skeleton_view *skeleton,
    float fps,const char *path,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && skeleton && path && *path,
                      "session, skeleton samples and output path are required");
        gemx::require(std::isfinite(fps) && fps>0 && fps<=1000,
                      "frames per second must be in (0,1000]");
        session->implementation->export_skeleton(*skeleton,fps,path);
    });
}

gemx_status gemx_export_skeleton_glb(gemx_session *session,const gemx_motion_view *motion,
    float fps,const char *path,char *error,uint64_t capacity){
    return gemx::boundary(error,capacity,[&]{
        gemx::require(session && session->implementation && motion && path && *path,
                      "session, motion and output path are required");
        gemx::require(std::isfinite(fps) && fps>0 && fps<=1000,"frames per second must be in (0,1000]");
        gemx::write_glb(session->implementation->skeleton(*motion),fps,path);
    });
}
