#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gemx {
struct skeleton_data {
    uint32_t frames=0;
    std::vector<float> positions;          // F,77,3 world
    std::vector<float> local_rotations;    // F,77,4 xyzw
    std::vector<float> local_translations; // F,77,3
    std::vector<int32_t> parents;           // 77, output indices
    std::vector<std::string> names;         // 77
};
void write_glb(const skeleton_data &,float fps,const std::string &path);
}
