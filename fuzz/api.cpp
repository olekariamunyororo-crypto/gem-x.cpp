#include "gemx.h"
#include <array>
#include <cstring>
#include <vector>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data,size_t size){
    if(size<4)return 0;
    char error[128]{};
    std::array<float,231+3+9+6+1024> input{};
    std::memcpy(input.data(),data+4,std::min(size-4,sizeof(input)));
    std::array<float,231> out{};float cliff[3]{};
    gemx_sequence_view view{1,input.data(),input.data()+231,input.data()+234,input.data()+249,input.data()+243};
    // Invalid frame counts must be rejected before accessing caller storage.
    if(data[1]&1)view.frames=data[2];
    gemx_preprocess_sequence(&view,out.data(),out.size(),cliff,3,error,sizeof(error));
    float matrix[9];gemx_rotation_6d_to_matrix(input.data(),matrix,error,sizeof(error));
    if(data[0]%4==0){
        std::array<uint8_t,32*32*3> rgb{};
        std::memcpy(rgb.data(),data,std::min(size,rgb.size()));
        gemx_rgb_frame f{rgb.data(),rgb.size(),uint32_t(data[1]),uint32_t(data[2]),uint64_t(data[3]),{input[0],input[1],input[2]}};
        std::vector<float> image(12*320*320);
        if(data[0]&4)gemx_vitpose_prepare_rgb(&f,image.data(),3*256*192,error,sizeof(error));
        else {float ratio;gemx_yolox_prepare_rgb(&f,image.data(),image.size(),&ratio,error,sizeof(error));}
    }
    std::array<float,585> norm{},mean{},dev{},motion{};
    std::memcpy(norm.data(),data,std::min(size,sizeof(norm)));
    dev.fill(1);
    gemx_denormalize_motion(norm.data(),1,mean.data(),dev.data(),motion.data(),error,sizeof(error));
    return 0;
}
