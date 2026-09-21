// Exercise production wire parsers without opening paths supplied by the input.
#define main gemx_pipeline_cli_main
#include "../tools/pipeline.cpp"
#undef main
#include <sstream>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data,size_t size){
    if(!size)return 0;
    std::string bytes(reinterpret_cast<const char *>(data+1),size-1);
    std::istringstream in(bytes);
    try{
        switch(data[0]%8){
        case 0:(void)parse_image(in);break;
        case 1:(void)parse_pose_token(in);break;
        case 2:(void)parse_pose(in);break;
        case 3:(void)parse_paths(in,"GEMIMGS1",120);break;
        case 4:(void)parse_sequence(in);break;
        case 5:(void)parse_paths(in,"GEMMAN01",GEMX_MAX_FRAMES);break;
        case 6:(void)number(bytes.c_str());break;
        case 7:(void)real(bytes.c_str());break;
        }
    }catch(const std::exception &){}
    return 0;
}
