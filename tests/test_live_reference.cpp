#include "gemx.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char **argv){
    try{
        if(argc!=4)throw std::runtime_error("usage: live-reference-test MODEL BACKEND FIXTURE");
        std::ifstream file(argv[3],std::ios::binary);char magic[8]{};std::array<uint32_t,4> header{};
        file.read(magic,8);file.read(reinterpret_cast<char *>(header.data()),16);
        const auto [frames,window,width,height]=header;
        if(std::memcmp(magic,"GEMLIVE1",8)||frames<2||frames>120||window<2||window>120||!width||!height)
            throw std::runtime_error("invalid live fixture");
        auto read=[&](uint32_t count,uint32_t channels){std::vector<float> v(uint64_t(count)*channels);
            if(!file.read(reinterpret_cast<char *>(v.data()),v.size()*4))throw std::runtime_error("truncated live fixture");return v;};
        auto kp=read(frames,231),boxes=read(frames,3),expected_x=read(frames-1,585),expected_cam=read(frames-1,3),
            expected_body=read(frames-1,228),expected_id=read(frames-1,45),expected_scale=read(frames-1,69),expected_orient=read(frames-1,3);
        if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing live fixture data");
        const char *backend=std::getenv("GEMX_TEST_BACKEND");if(!backend)backend="CPU";
        gemx_session_config cfg{argv[1],argv[2],backend,nullptr,0,8,32};gemx_session *raw=nullptr;char error[512]{};
        if(gemx_session_create_live(&cfg,&raw,error,sizeof(error))!=GEMX_OK)throw std::runtime_error(error);
        std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)> session(raw,gemx_session_destroy);
        std::vector<float> K(frames*9),angular(frames*6);
        for(uint32_t f=0;f<frames;++f){K[f*9]=K[f*9+4]=float(std::max(width,height));K[f*9+2]=width*.5f;
            K[f*9+5]=height*.5f;K[f*9+8]=1;angular[f*6]=angular[f*6+4]=1;}
        std::array<float,7> maxima{};
        auto compare=[&](const float *actual,const float *expected,uint32_t size,int index){
            for(uint32_t j=0;j<size;++j){if(!std::isfinite(actual[j]))throw std::runtime_error("nonfinite live output");
                maxima[index]=std::max(maxima[index],std::abs(actual[j]-expected[j]));}};
        for(uint32_t end=2;end<=frames;++end){
            const uint32_t length=std::min(end,window),start=end-length,target=end-2;
            gemx_sequence_view input{length,kp.data()+start*231,boxes.data()+start*3,K.data()+start*9,nullptr,angular.data()+start*6};
            std::vector<float> x(length*585),cam(length*3),body(length*228),identity(length*45),scales(length*69),
                oc(length*3),tc(length*3),ow(length*3),tw(length*3);
            if(gemx_infer(raw,&input,x.data(),x.size(),cam.data(),cam.size(),error,sizeof(error))!=GEMX_OK)throw std::runtime_error(error);
            gemx_motion_view motion{length,body.data(),identity.data(),scales.data(),oc.data(),tc.data(),ow.data(),tw.data()};
            if(gemx_decode_predictions(raw,&input,x.data(),x.size(),cam.data(),cam.size(),&motion,error,sizeof(error))!=GEMX_OK)
                throw std::runtime_error(error);
            compare(x.data()+(length-1)*585,expected_x.data()+target*585,585,0);
            compare(cam.data()+(length-1)*3,expected_cam.data()+target*3,3,1);
            compare(body.data()+(length-1)*228,expected_body.data()+target*228,228,2);
            compare(identity.data()+(length-1)*45,expected_id.data()+target*45,45,3);
            compare(scales.data()+(length-1)*69,expected_scale.data()+target*69,69,4);
            compare(ow.data()+(length-1)*3,expected_orient.data()+target*3,3,5);
        }
        for(int i=0;i<6;++i){std::printf("live component %d max=%g\n",i,maxima[i]);
            // Float32 acos near identity amplifies last-bit differences in
            // upstream's yaw rollout; 0.004 rad bounds that separately.
            if(maxima[i]>(i==5?.004f:1e-4f))throw std::runtime_error("live reference tolerance exceeded (requires strict F32)");}
        return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
