#define main gemx_pipeline_cli_main
#include "../tools/pipeline.cpp"
#undef main
#include <sstream>
int main(){
    bool rejected=false;
    try{(void)real("");}catch(const std::invalid_argument &){rejected=true;}
    if(!rejected||real("1.25")!=1.25f)throw std::runtime_error("numeric parsing contract");
    std::ostringstream out;write(out,"GEMMAN01",8);
    uint32_t count=1,length=5;write(out,&count);write(out,&length);write(out,"a.bin",5);
    std::istringstream valid(out.str());
    if(parse_paths(valid,"GEMMAN01",4096)!=std::vector<std::string>{"a.bin"})throw std::runtime_error("path roundtrip");
    std::istringstream trailing(out.str()+"x");rejected=false;
    try{(void)parse_paths(trailing,"GEMMAN01",4096);}catch(const std::invalid_argument &){rejected=true;}
    if(!rejected)throw std::runtime_error("trailing manifest accepted");
    for(const auto &payload:{std::string("S3DIMG01"),std::string("S3DOUT01"),std::string("GEMPOSE2")}){
        std::istringstream in(payload);rejected=false;
        try{if(payload=="S3DIMG01")parse_image(in);else if(payload=="S3DOUT01")parse_pose_token(in);else parse_pose(in);}
        catch(const std::invalid_argument &){rejected=true;}
        if(!rejected)throw std::runtime_error("truncated input accepted");
    }
}
