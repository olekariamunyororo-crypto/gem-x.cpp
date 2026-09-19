#include "gemx.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char **argv){
    try{
        if(argc!=4)throw std::runtime_error("usage: contacts-test MODEL CPU_BACKEND FIXTURE");
        std::ifstream file(argv[3],std::ios::binary);char magic[8]{};uint32_t frames=0;
        file.read(magic,8);file.read(reinterpret_cast<char *>(&frames),4);
        if(std::memcmp(magic,"GEMCONT1",8)||frames<1||frames>120)throw std::runtime_error("invalid contact fixture");
        auto read=[&](int width){std::vector<float> v(uint64_t(frames)*width);
            if(!file.read(reinterpret_cast<char *>(v.data()),v.size()*4))throw std::runtime_error("truncated contact fixture");return v;};
        auto body=read(228),identity=read(45),scales=read(69),orient=read(3),translation=read(3),
            contacts=read(6),expected_body=read(228),expected_translation=read(3);
        if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing fixture data");
        gemx_session_config cfg{argv[1],argv[2],"CPU",nullptr,0,8,1};char error[512]{};gemx_session *raw=nullptr;
        if(gemx_session_create(&cfg,&raw,error,sizeof(error))!=GEMX_OK)throw std::runtime_error(error);
        std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)> session(raw,gemx_session_destroy);
        gemx_motion_view motion{frames,body.data(),identity.data(),scales.data(),orient.data(),translation.data(),orient.data(),translation.data()};
        if(gemx_refine_contacts(raw,&motion,contacts.data(),contacts.size(),error,sizeof(error))!=GEMX_OK)throw std::runtime_error(error);
        auto check=[](const auto &actual,const auto &expected,float limit,const char *label){
            float maximum=0;double mean=0;
            for(size_t i=0;i<actual.size();++i){
                if(!std::isfinite(actual[i]))throw std::runtime_error("nonfinite contact output");
                float d=std::abs(actual[i]-expected[i]);maximum=std::max(maximum,d);mean+=d;
            }
            std::printf("%s max=%g mean=%g\n",label,maximum,mean/actual.size());
            if(maximum>limit)throw std::runtime_error("contact result differs from actual upstream reference");
        };
        check(body,expected_body,1e-3f,"contact body axis-angle");
        check(translation,expected_translation,1e-5f,"contact translation metres");
        auto saved=body;contacts.back()=std::numeric_limits<float>::quiet_NaN();
        if(gemx_refine_contacts(raw,&motion,contacts.data(),contacts.size(),error,sizeof(error))!=GEMX_INVALID_ARGUMENT || saved!=body)
            throw std::runtime_error("nonfinite contact input must fail without mutating motion");
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
