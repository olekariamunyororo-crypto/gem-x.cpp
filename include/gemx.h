#ifndef GEMX_H
#define GEMX_H
#include <stdint.h>
#if defined(_WIN32)
# if defined(GEMX_BUILDING_LIBRARY)
#  define GEMX_API __declspec(dllexport)
# else
#  define GEMX_API __declspec(dllimport)
# endif
#elif defined(__GNUC__)
# define GEMX_API __attribute__((visibility("default")))
#else
# define GEMX_API
#endif
#ifdef __cplusplus
extern "C" {
#endif

typedef enum gemx_status {
    GEMX_OK=0,
    GEMX_INVALID_ARGUMENT=1,
    GEMX_OUT_OF_MEMORY=2,
    GEMX_RUNTIME_ERROR=3
} gemx_status;

#define GEMX_SOMA_JOINTS UINT32_C(77)
#define GEMX_BODY_FEATURE_DIM UINT32_C(1024)
#define GEMX_MOTION_DIM UINT32_C(585)
#define GEMX_MAX_FRAMES UINT32_C(4096)

/* Structure-of-arrays sequence input. Every pointer covers the documented
 * count and is borrowed only for the duration of the call. Inputs are F32:
 * keypoints [frames,77,3] pixel x/y/confidence, boxes [frames,3] cx/cy/size,
 * intrinsics [frames,3,3], Body features [frames,1024], and camera angular
 * velocity [frames,6]. This preprocessing API does not run learned inference. */
typedef struct gemx_sequence_view {
    uint32_t frames;
    const float *keypoints;
    const float *boxes;
    const float *intrinsics;
    const float *body_features;
    const float *camera_angular_velocity;
} gemx_sequence_view;

typedef struct gemx_session gemx_session;
typedef struct gemx_vitpose gemx_vitpose;
typedef struct gemx_yolox gemx_yolox;

typedef struct gemx_session_config {
    const char *model_path;
    /* Explicit backend module, or a GGML backend directory. A directory makes
     * GGML score and load the best compatible CPU build variant at runtime. */
    const char *backend_module;
    const char *backend_name; /* "CPU" or "Vulkan" */
    const char *expected_device_description; /* optional exact match */
    uint32_t device_index;
    uint32_t threads;
    uint32_t graph_cache_capacity;
} gemx_session_config;

typedef struct gemx_rgb_frame {
    const uint8_t *rgb;
    uint64_t capacity;
    uint32_t width;
    uint32_t height;
    uint64_t row_stride;
    float box[3]; /* cx, cy and square size in source-image pixels */
} gemx_rgb_frame;

typedef struct gemx_detection {
    float box[4]; /* source-image x0,y0,x1,y1 */
    float score;
} gemx_detection;

typedef struct gemx_profile {
    uint64_t calls;
    uint64_t frames;
    uint64_t preprocessing_ns;
    uint64_t graph_build_ns;
    uint64_t upload_ns;
    uint64_t inference_ns;
    uint64_t download_ns;
    uint64_t graph_cache_hits;
    uint64_t graph_cache_misses;
} gemx_profile;

typedef struct gemx_motion_view {
    uint32_t frames;
    float *body_pose;          /* [frames,76,3] axis-angle */
    float *identity_coeffs;    /* [frames,45] */
    float *scale_params;       /* [frames,69], global scale clamped to .7..1 */
    float *global_orient_camera; /* [frames,3] axis-angle */
    float *translation_camera;   /* [frames,3] metres */
    float *global_orient_world;  /* [frames,3] axis-angle */
    float *translation_world;    /* [frames,3] metres, first frame at origin */
} gemx_motion_view;

#define GEMX_SOMA_RIG_JOINTS UINT32_C(77)
typedef struct gemx_skeleton_view {
    uint32_t frames;
    float *joint_positions; /* [frames,77,3] world-space metres */
    float *local_rotations; /* [frames,77,4] glTF xyzw quaternions */
    int32_t *parents;       /* optional [77], -1 for Hips */
    float *local_translations; /* optional [frames,77,3] metres */
} gemx_skeleton_view;

/* Writes normalized keypoints [frames,77,3] and BEDLAM/CLIFF box conditions
 * [frames,3]. Confidence becomes zero outside the square person box. */
GEMX_API gemx_status gemx_preprocess_sequence(const gemx_sequence_view *input,
    float *normalized_keypoints,uint64_t normalized_count,
    float *box_conditions,uint64_t box_condition_count,
    char *error,uint64_t error_capacity);

/* Convert one 6D rotation (the first two rows used by upstream) into a
 * row-major 3x3 rotation matrix using Gram-Schmidt normalization. */
GEMX_API gemx_status gemx_rotation_6d_to_matrix(const float rotation[6],
    float matrix[9],char *error,uint64_t error_capacity);

/* Denormalize raw GEM motion using fixed 585-value mean/std arrays. */
GEMX_API gemx_status gemx_denormalize_motion(const float *normalized,
    uint32_t frames,const float *mean,const float *stddev,
    float *motion,char *error,uint64_t error_capacity);

/* Create a single-backend native denoiser. No backend or device is substituted. */
GEMX_API gemx_status gemx_session_create(const gemx_session_config *config,
    gemx_session **session,char *error,uint64_t error_capacity);
GEMX_API void gemx_session_destroy(gemx_session *session);

/* Match the two outputs of NVIDIA's released regression ONNX graph. Sequences
 * up to 4096 frames are returned in full and evaluated as consecutive windows
 * of at most 120 frames. Exact upstream parity is covered through 120 frames;
 * upstream uses an overlapping local-attention policy for longer sequences.
 * Outputs are F32 arrays [frames,585] and [frames,3]. */
GEMX_API gemx_status gemx_infer(gemx_session *session,
    const gemx_sequence_view *input,float *pred_x,uint64_t pred_x_count,
    float *pred_camera,uint64_t pred_camera_count,
    char *error,uint64_t error_capacity);

/* Infer and apply the published SOMA-v2 denormalization, rotation decoding,
 * weak-perspective camera conversion and global camera-motion rollout. */
GEMX_API gemx_status gemx_infer_motion(gemx_session *session,
    const gemx_sequence_view *input,const gemx_motion_view *output,
    char *error,uint64_t error_capacity);

/* Decode caller-supplied raw denoiser/camera predictions using the same
 * published postprocessing as gemx_infer_motion. */
GEMX_API gemx_status gemx_decode_predictions(gemx_session *session,
    const gemx_sequence_view *input,const float *pred_x,uint64_t pred_x_count,
    const float *pred_camera,uint64_t pred_camera_count,
    const gemx_motion_view *output,char *error,uint64_t error_capacity);

GEMX_API gemx_status gemx_build_skeleton(gemx_session *session,
    const gemx_motion_view *motion,const gemx_skeleton_view *skeleton,
    char *error,uint64_t error_capacity);

/* Export already-built SOMA samples. Parents must be the session's SOMA
 * topology; local translations/rotations are required. */
GEMX_API gemx_status gemx_export_skeleton_samples_glb(gemx_session *session,
    const gemx_skeleton_view *skeleton,float frames_per_second,const char *path,
    char *error,uint64_t error_capacity);

/* Write a standards-compliant binary glTF containing the 77-node animated
 * SOMA hierarchy. The export is skeleton-only and uses world motion. */
GEMX_API gemx_status gemx_export_skeleton_glb(gemx_session *session,
    const gemx_motion_view *motion,float frames_per_second,const char *path,
    char *error,uint64_t error_capacity);

GEMX_API gemx_status gemx_session_get_profile(const gemx_session *session,
    gemx_profile *profile,char *error,uint64_t error_capacity);
GEMX_API const char *gemx_session_device(const gemx_session *session);

/* Native NVIDIA GEM-X ViTPose SOMA-77 observation model. The normalized API
 * accepts ImageNet-normalized NCHW crops [batch,3,256,192], with batch 1..8,
 * and returns heatmaps [batch,77,64,48]. The RGB API performs the published
 * square crop, horizontal flip test, UDP peak refinement and source-image
 * coordinate reconstruction. It processes 1..4 frames in one bounded graph. */
GEMX_API gemx_status gemx_vitpose_create(const gemx_session_config *config,
    gemx_vitpose **model,char *error,uint64_t error_capacity);
GEMX_API void gemx_vitpose_destroy(gemx_vitpose *model);
GEMX_API const char *gemx_vitpose_device(const gemx_vitpose *model);
GEMX_API gemx_status gemx_vitpose_prepare_rgb(const gemx_rgb_frame *frame,
    float *normalized_image,uint64_t normalized_count,
    char *error,uint64_t error_capacity);
GEMX_API gemx_status gemx_vitpose_infer_normalized(gemx_vitpose *model,
    const float *images,uint32_t batch,float *heatmaps,uint64_t heatmap_count,
    char *error,uint64_t error_capacity);
GEMX_API gemx_status gemx_vitpose_infer_rgb(gemx_vitpose *model,
    const gemx_rgb_frame *frames,uint32_t frame_count,
    float *keypoints,uint64_t keypoint_count,char *error,uint64_t error_capacity);

/* Native YOLOX-X HumanArt person detector used by NVIDIA GEM-X. The RGB API
 * performs the published top-left 640-square letterbox and BGR conversion.
 * Detection decodes only COCO class zero (person), then applies greedy NMS. */
GEMX_API gemx_status gemx_yolox_create(const gemx_session_config *config,
    gemx_yolox **model,char *error,uint64_t error_capacity);
GEMX_API void gemx_yolox_destroy(gemx_yolox *model);
GEMX_API const char *gemx_yolox_device(const gemx_yolox *model);
GEMX_API gemx_status gemx_yolox_prepare_rgb(const gemx_rgb_frame *frame,
    float *focused_image,uint64_t focused_count,float *resize_ratio,
    char *error,uint64_t error_capacity);
GEMX_API gemx_status gemx_yolox_detect(gemx_yolox *model,
    const gemx_rgb_frame *frame,float score_threshold,float nms_threshold,
    gemx_detection *detections,uint32_t detection_capacity,uint32_t *detection_count,
    char *error,uint64_t error_capacity);

#ifdef __cplusplus
}
#endif
#endif
