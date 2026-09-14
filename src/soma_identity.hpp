#pragma once
#include <cstdint>
#include <vector>

namespace gemx {

struct soma_identity_constants {
    std::vector<float> mhr_offsets, mhr_prerotations, mhr_parameter_matrix;
    std::vector<float> mhr_inverse_bind, mhr_skin_weights;
    std::vector<float> mhr_shape_vectors, mhr_base_shape;
    std::vector<int32_t> mhr_parents, mhr_skin_joints, mhr_skin_vertices, mhr_faces;
    std::vector<int32_t> transfer_face_ids;
    std::vector<float> transfer_barycentric;
    std::vector<int32_t> rbf_crow, rbf_columns;
    std::vector<float> rbf_values, bind_world;
    std::vector<int32_t> soma_parents;
    std::vector<int32_t> rotation_crow, rotation_vertices;
    std::vector<float> rotation_reference;
};

struct soma_identity_rig {
    std::vector<float> fitted_positions; // [78,3], meters
    std::vector<float> fitted_rotations; // [78,3,3], world
    std::vector<float> local_offsets;    // [78,3], relative to fitted parent
};

soma_identity_rig fit_soma_identity(const soma_identity_constants &constants,
                                    const float *identity45,
                                    const float *scale68,
                                    float global_scale);
}
