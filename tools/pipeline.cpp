#include "gemx.h"
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using session_ptr=std::unique_ptr<gemx_session,decltype(&gemx_session_destroy)>;
using vitpose_ptr=std::unique_ptr<gemx_vitpose,decltype(&gemx_vitpose_destroy)>;
using yolox_ptr=std::unique_ptr<gemx_yolox,decltype(&gemx_yolox_destroy)>;
struct live_delete {void operator()(gemx_live *value) const{gemx_live_destroy(value);}};
using live_ptr=std::unique_ptr<gemx_live,live_delete>;
template<class T> void read(std::istream &in,T *value,size_t count=1){
    if(!in.read(reinterpret_cast<char *>(value),sizeof(T)*count))throw std::invalid_argument("truncated input");
}
template<class T> void write(std::ostream &out,const T *value,size_t count=1){
    if(!out.write(reinterpret_cast<const char *>(value),sizeof(T)*count))throw std::runtime_error("output write failed");
}
uint32_t number(const char *text){
    uint32_t value=0;auto end=text+std::strlen(text);auto result=std::from_chars(text,end,value);
    if(result.ec!=std::errc{}||result.ptr!=end)throw std::invalid_argument("invalid integer");
    return value;
}
float real(const char *text){
    char *end=nullptr;errno=0;float value=std::strtof(text,&end);
    if(errno||end!=text+std::strlen(text)||!std::isfinite(value))throw std::invalid_argument("invalid number");
    return value;
}
void api(gemx_status status,const char *message){if(status!=GEMX_OK)throw std::runtime_error(message);}

struct image_data {
    uint32_t width=0,height=0,stride=0;
    std::array<float,4> box{},camera{};
    std::vector<uint8_t> rgb;
};
image_data load_image(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open packed image");
    std::array<char,8> magic{};read(in,magic.data(),magic.size());
    if(std::string(magic.data(),magic.size())!="S3DIMG01")throw std::invalid_argument("wrong packed image magic");
    image_data value;read(in,&value.width);read(in,&value.height);read(in,&value.stride);
    read(in,value.box.data(),4);read(in,value.camera.data(),4);
    if(value.width<8||value.height<8||value.width>32766||value.height>32766||
       uint64_t(value.width)*value.height>16000000||value.stride!=value.width*3)
        throw std::invalid_argument("invalid packed image dimensions");
    value.rgb.resize(uint64_t(value.stride)*value.height);read(in,value.rgb.data(),value.rgb.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing packed image data");
    return value;
}

std::array<float,1024> load_pose_token(const std::string &path){
    for(unsigned attempt=0;attempt<600000;++attempt){
        std::error_code error;
        if(std::filesystem::file_size(path,error)>=400000&&!error)break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::ifstream in(path,std::ios::binary);
    if(!in)throw std::runtime_error("body result did not become available");
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!="S3DOUT01")throw std::invalid_argument("wrong body result magic");
    uint32_t tensors=0;read(in,&tensors);if(tensors<1||tensors>64)throw std::invalid_argument("invalid body tensor count");
    std::array<float,1024> result{};
    for(uint32_t tensor=0;tensor<tensors;++tensor){
        uint32_t name_size=0,type=0,rank=0;uint64_t elements=0;read(in,&name_size);
        if(name_size<1||name_size>64)throw std::invalid_argument("invalid tensor name");
        std::string name(name_size,'\0');read(in,name.data(),name.size());
        read(in,&type);read(in,&rank);read(in,&elements);
        if(rank>8||elements>(1ull<<28))throw std::invalid_argument("invalid tensor descriptor");
        uint64_t product=1;std::array<uint64_t,8> dimensions{};
        for(uint32_t i=0;i<rank;++i){read(in,&dimensions[i]);if(!dimensions[i]||dimensions[i]>(1ull<<28)/product)throw std::invalid_argument("invalid tensor shape");product*=dimensions[i];}
        if(product!=elements)throw std::invalid_argument("tensor element count mismatch");
        if(name=="pose_token"){
            if(type!=1||rank!=2||dimensions[0]!=1||dimensions[1]!=1024)
                throw std::invalid_argument("invalid GEM-X pose token");
            read(in,result.data(),result.size());return result;
        }else{
            in.seekg(static_cast<std::streamoff>(elements*4),std::ios::cur);
            if(!in)throw std::invalid_argument("truncated tensor payload");
        }
    }
    throw std::invalid_argument("body result lacks GEM-X pose token");
}

struct pose_sample {
    std::array<float,77*3> positions{},local_translations{},keypoints{};
    std::array<float,77*4> rotations{};
    std::array<int32_t,77> parents{};
    std::array<float,3> camera{};
};
void save_pose(const std::string &path,const pose_sample &value){
    std::ofstream out(path,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot create GEM-X pose");
    write(out,"GEMPOSE1",8);write(out,value.positions.data(),value.positions.size());
    write(out,value.rotations.data(),value.rotations.size());
    write(out,value.local_translations.data(),value.local_translations.size());
    write(out,value.parents.data(),value.parents.size());write(out,value.camera.data(),3);
    write(out,value.keypoints.data(),value.keypoints.size());
}
pose_sample load_pose(const std::string &path){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot open GEM-X pose");
    std::array<char,8> magic{};read(in,magic.data(),8);
    if(std::string(magic.data(),8)!="GEMPOSE1")throw std::invalid_argument("wrong GEM-X pose magic");
    pose_sample value;read(in,value.positions.data(),value.positions.size());
    read(in,value.rotations.data(),value.rotations.size());
    read(in,value.local_translations.data(),value.local_translations.size());
    read(in,value.parents.data(),value.parents.size());read(in,value.camera.data(),3);
    read(in,value.keypoints.data(),value.keypoints.size());
    if(in.peek()!=std::char_traits<char>::eof())throw std::invalid_argument("trailing GEM-X pose data");
    return value;
}

std::string field(bool boundary=false){
    std::array<uint8_t,4> bytes{};size_t got=std::fread(bytes.data(),1,4,stdin);
    if(!got&&boundary&&std::feof(stdin))return {};
    if(got!=4)throw std::invalid_argument("truncated worker request");
    uint32_t size=uint32_t(bytes[0])|uint32_t(bytes[1])<<8|uint32_t(bytes[2])<<16|uint32_t(bytes[3])<<24;
    if(!size||size>4096)throw std::invalid_argument("invalid worker field");
    std::string value(size,'\0');if(std::fread(value.data(),1,size,stdin)!=size||value.find('\0')!=std::string::npos)
        throw std::invalid_argument("invalid worker field data");
    return value;
}
gemx_session_config config(const char *model,const char *module,const char *backend,
                           const char *description,uint32_t device,uint32_t threads,uint32_t cache){
    return {model,module,backend,std::strcmp(description,"-")?description:"",device,threads,cache};
}

float iou(const std::array<float,4> &a,const std::array<float,4> &b){
    const float x0=std::max(a[0],b[0]),y0=std::max(a[1],b[1]);
    const float x1=std::min(a[2],b[2]),y1=std::min(a[3],b[3]);
    const float intersection=std::max(0.f,x1-x0)*std::max(0.f,y1-y0);
    const float aa=std::max(0.f,a[2]-a[0])*std::max(0.f,a[3]-a[1]);
    const float ab=std::max(0.f,b[2]-b[0])*std::max(0.f,b[3]-b[1]);
    return intersection/(aa+ab-intersection+1e-7f);
}

// The state, noise matrices and two-stage association match the compact
// ByteTrack implementation released by NVIDIA with GEM-X.
struct kalman_track {
    std::array<double,8> x{};std::array<double,64> P{};
    int id=0,hits=1,lost=0;float score=0;
    kalman_track(const std::array<float,4> &box,float confidence,int identity):id(identity),score(confidence){
        const double w=box[2]-box[0],h=box[3]-box[1];
        x={(box[0]+box[2])*.5,(box[1]+box[3])*.5,w/(h+1e-7),h,0,0,0,0};
        for(int i=0;i<8;++i)P[i*8+i]=i<4?10.:1000.;
    }
    std::array<float,4> box() const{
        const double h=x[3],w=x[2]*h;
        return {float(x[0]-w*.5),float(x[1]-h*.5),float(x[0]+w*.5),float(x[1]+h*.5)};
    }
    void predict(){
        for(int i=0;i<4;++i)x[i]+=x[i+4];
        auto old=P;
        for(int r=0;r<8;++r)for(int c=0;c<8;++c){
            double value=old[r*8+c];
            if(r<4)value+=old[(r+4)*8+c];
            if(c<4)value+=old[r*8+c+4];
            if(r<4&&c<4)value+=old[(r+4)*8+c+4];
            P[r*8+c]=value;
        }
        for(int i=0;i<8;++i)P[i*8+i]+=i<4?1.:.01;
        ++lost;
    }
    void update(const std::array<float,4> &box,float confidence){
        const double w=box[2]-box[0],h=box[3]-box[1];
        std::array<double,4> z{(box[0]+box[2])*.5,(box[1]+box[3])*.5,w/(h+1e-7),h};
        double augmented[4][8]{};
        for(int r=0;r<4;++r){
            for(int c=0;c<4;++c)augmented[r][c]=P[r*8+c]+(r==c?(r==2?10.:1.):0.);
            augmented[r][4+r]=1.;
        }
        for(int column=0;column<4;++column){
            int pivot=column;for(int row=column+1;row<4;++row)
                if(std::abs(augmented[row][column])>std::abs(augmented[pivot][column]))pivot=row;
            if(std::abs(augmented[pivot][column])<1e-12)throw std::runtime_error("singular ByteTrack covariance");
            if(pivot!=column)for(int c=0;c<8;++c)std::swap(augmented[pivot][c],augmented[column][c]);
            const double scale=augmented[column][column];for(double &v:augmented[column])v/=scale;
            for(int row=0;row<4;++row)if(row!=column){const double factor=augmented[row][column];
                for(int c=0;c<8;++c)augmented[row][c]-=factor*augmented[column][c];}
        }
        double K[8][4]{};
        for(int r=0;r<8;++r)for(int c=0;c<4;++c)for(int k=0;k<4;++k)
            K[r][c]+=P[r*8+k]*augmented[k][4+c];
        std::array<double,4> residual{};for(int i=0;i<4;++i)residual[i]=z[i]-x[i];
        for(int r=0;r<8;++r)for(int c=0;c<4;++c)x[r]+=K[r][c]*residual[c];
        auto old=P;for(int r=0;r<8;++r)for(int c=0;c<8;++c){
            double value=old[r*8+c];for(int k=0;k<4;++k)value-=K[r][k]*old[k*8+c];P[r*8+c]=value;
        }
        ++hits;lost=0;score=confidence;
    }
};

std::vector<std::pair<int,int>> hungarian(const std::vector<std::vector<double>> &cost){
    if(cost.empty()||cost[0].empty())return {};
    const int rows=int(cost.size()),columns=int(cost[0].size());const bool transpose=rows>columns;
    const int n=transpose?columns:rows,m=transpose?rows:columns;
    auto value=[&](int row,int column){return transpose?cost[column][row]:cost[row][column];};
    std::vector<double> u(n+1),v(m+1);std::vector<int> p(m+1),way(m+1);
    for(int i=1;i<=n;++i){
        p[0]=i;int j0=0;std::vector<double> minimum(m+1,std::numeric_limits<double>::infinity());
        std::vector<bool> used(m+1);
        do{
            used[j0]=true;const int i0=p[j0];double delta=std::numeric_limits<double>::infinity();int j1=0;
            for(int j=1;j<=m;++j)if(!used[j]){const double current=value(i0-1,j-1)-u[i0]-v[j];
                if(current<minimum[j]){minimum[j]=current;way[j]=j0;}if(minimum[j]<delta){delta=minimum[j];j1=j;}}
            for(int j=0;j<=m;++j)if(used[j]){u[p[j]]+=delta;v[j]-=delta;}else minimum[j]-=delta;
            j0=j1;
        }while(p[j0]!=0);
        do{const int j1=way[j0];p[j0]=p[j1];j0=j1;}while(j0);
    }
    std::vector<std::pair<int,int>> result;
    for(int j=1;j<=m;++j)if(p[j])result.push_back(transpose?std::pair{j-1,p[j]-1}:std::pair{p[j]-1,j-1});
    return result;
}

struct byte_tracker {
    std::vector<kalman_track> tracks;int next_id=0,target=-1;std::array<float,4> last{};
    std::deque<std::array<float,4>> smoothing;
    void reset(){tracks.clear();next_id=0;target=-1;smoothing.clear();last={};}
    struct association {std::vector<std::pair<int,int>> matched;std::vector<int> tracks,detections;};
    association match(const std::vector<int> &track_indices,const std::vector<gemx_detection> &detections,
                      const std::vector<int> &detection_indices,float threshold){
        association result;result.tracks=track_indices;result.detections=detection_indices;
        if(track_indices.empty()||detection_indices.empty())return result;
        std::vector<std::vector<double>> cost(track_indices.size(),std::vector<double>(detection_indices.size()));
        for(size_t i=0;i<track_indices.size();++i)for(size_t j=0;j<detection_indices.size();++j)
            cost[i][j]=1.-iou(tracks[track_indices[i]].box(),{detections[detection_indices[j]].box[0],detections[detection_indices[j]].box[1],detections[detection_indices[j]].box[2],detections[detection_indices[j]].box[3]});
        std::vector<bool> used_track(track_indices.size()),used_detection(detection_indices.size());
        for(auto [row,column]:hungarian(cost))if(1.-cost[row][column]>=threshold){
            result.matched.emplace_back(track_indices[row],detection_indices[column]);used_track[row]=true;used_detection[column]=true;
        }
        result.tracks.clear();result.detections.clear();
        for(size_t i=0;i<track_indices.size();++i)if(!used_track[i])result.tracks.push_back(track_indices[i]);
        for(size_t i=0;i<detection_indices.size();++i)if(!used_detection[i])result.detections.push_back(detection_indices[i]);
        return result;
    }
    std::array<float,4> update(const gemx_detection *values,uint32_t count,const std::array<float,4> &selection,
                               uint32_t width,uint32_t height){
        std::vector<gemx_detection> detections(values,values+count);for(auto &track:tracks)track.predict();
        std::vector<int> all_tracks(tracks.size()),high,low;std::iota(all_tracks.begin(),all_tracks.end(),0);
        for(uint32_t i=0;i<count;++i)if(values[i].score>=.5f)high.push_back(i);else if(values[i].score>.1f)low.push_back(i);
        auto first=match(all_tracks,detections,high,.3f);
        for(auto [track,detection]:first.matched)tracks[track].update({values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score);
        auto second=match(first.tracks,detections,low,.3f);
        for(auto [track,detection]:second.matched)tracks[track].update({values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score);
        for(int detection:first.detections)tracks.emplace_back(
            std::array<float,4>{values[detection].box[0],values[detection].box[1],values[detection].box[2],values[detection].box[3]},values[detection].score,next_id++);
        tracks.erase(std::remove_if(tracks.begin(),tracks.end(),[](const auto &track){return track.lost>30;}),tracks.end());
        kalman_track *chosen=nullptr;for(auto &track:tracks)if(track.id==target)chosen=&track;
        if(!chosen){
            const auto &reference=target<0?selection:last;float best=-1;
            for(auto &track:tracks)if(track.lost==0){const float score=iou(track.box(),reference);if(score>best){best=score;chosen=&track;}}
            if(chosen&&best>0){target=chosen->id;smoothing.clear();}else chosen=nullptr;
        }
        std::array<float,4> box=chosen?chosen->box():(target<0?selection:last);
        for(int axis=0;axis<4;++axis){const float limit=axis%2?float(height-1):float(width-1);box[axis]=std::clamp(box[axis],0.f,limit);}
        if(chosen&&chosen->lost==0){smoothing.push_back(box);if(smoothing.size()>3)smoothing.pop_front();
            box={};for(const auto &item:smoothing)for(int axis=0;axis<4;++axis)box[axis]+=item[axis]/smoothing.size();}
        if(box[2]-box[0]<8||box[3]-box[1]<8)box=selection;last=box;return box;
    }
};

int worker(int argc,char **argv){
    if(argc!=11)throw std::invalid_argument("usage: gemx-pipeline --worker DENOISER VITPOSE YOLOX MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS CONTEXT");
    const uint32_t device=number(argv[7]),threads=number(argv[9]),context=number(argv[10]);
    char error[512]{};
    auto den_cfg=config(argv[2],argv[5],argv[6],argv[8],device,threads,8);
    gemx_session *raw_session=nullptr;api(gemx_session_create(&den_cfg,&raw_session,error,sizeof(error)),error);
    session_ptr session(raw_session,gemx_session_destroy);
    auto pose_cfg=config(argv[3],argv[5],argv[6],argv[8],device,threads,4);
    gemx_vitpose *raw_pose=nullptr;api(gemx_vitpose_create(&pose_cfg,&raw_pose,error,sizeof(error)),error);
    vitpose_ptr pose(raw_pose,gemx_vitpose_destroy);
    auto detector_cfg=config(argv[4],argv[5],argv[6],argv[8],device,threads,1);
    gemx_yolox *raw_detector=nullptr;api(gemx_yolox_create(&detector_cfg,&raw_detector,error,sizeof(error)),error);
    yolox_ptr detector(raw_detector,gemx_yolox_destroy);byte_tracker tracker;
    gemx_live *raw_live=nullptr;api(gemx_live_create(session.get(),context,&raw_live,error,sizeof(error)),error);
    live_ptr live(raw_live);std::string current_stream;
    std::fputs("READY\n",stdout);std::fflush(stdout);
    for(;;){
        auto stream=field(true);if(stream.empty())break;
        if(stream.size()>128)throw std::invalid_argument("stream identifier is too long");
        auto image_path=field(),body_path=field(),output_path=field();
        if(stream!=current_stream){gemx_live_reset(live.get());tracker.reset();current_stream=stream;}
        image_data image;
        try{image=load_image(image_path);}catch(const std::exception &e){throw std::runtime_error(std::string("packed image: ")+e.what());}
        gemx_rgb_frame detector_frame{image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,{}};
        std::array<gemx_detection,100> detections{};uint32_t detection_count=0;
        api(gemx_yolox_detect(detector.get(),&detector_frame,.1f,.65f,detections.data(),detections.size(),
                              &detection_count,error,sizeof(error)),error);
        std::array<float,4> selected=image.box;
        const auto tracked=tracker.update(detections.data(),detection_count,selected,image.width,image.height);
        std::printf("BOX %.9g %.9g %.9g %.9g\n",tracked[0],tracked[1],tracked[2],tracked[3]);std::fflush(stdout);
        const float width=tracked[2]-tracked[0],height=tracked[3]-tracked[1];
        const float size=std::max(height,width/.75f)*1.2f;
        gemx_rgb_frame frame{image.rgb.data(),image.rgb.size(),image.width,image.height,image.stride,
                             {(tracked[0]+tracked[2])*.5f,(tracked[1]+tracked[3])*.5f,size}};
        pose_sample sample;api(gemx_vitpose_infer_rgb(pose.get(),&frame,1,sample.keypoints.data(),
                                                     sample.keypoints.size(),error,sizeof(error)),error);
        std::array<float,1024> token{};
        try{token=load_pose_token(body_path);}catch(const std::exception &e){throw std::runtime_error(std::string("body token: ")+e.what());}
        std::array<float,9> K={image.camera[0],0,image.camera[2],0,image.camera[1],image.camera[3],0,0,1};
        std::array<float,3> box={frame.box[0],frame.box[1],frame.box[2]};
        std::array<float,6> angular{};
        gemx_sequence_view observation{1,sample.keypoints.data(),box.data(),K.data(),token.data(),angular.data()};
        std::array<float,585> pred{};std::array<float,3> pred_camera{};
        api(gemx_live_push(live.get(),&observation,pred.data(),pred_camera.data(),error,sizeof(error)),error);
        std::array<float,76*3> body{};std::array<float,45> identity{};std::array<float,69> scales{};
        std::array<float,3> orient_camera{},translation_camera{},orient_world{},translation_world{};
        gemx_motion_view motion{1,body.data(),identity.data(),scales.data(),orient_camera.data(),
                                translation_camera.data(),orient_world.data(),translation_world.data()};
        api(gemx_decode_predictions(session.get(),&observation,pred.data(),pred.size(),pred_camera.data(),
                                    pred_camera.size(),&motion,error,sizeof(error)),error);
        gemx_skeleton_view skeleton{1,sample.positions.data(),sample.rotations.data(),sample.parents.data(),
                                    sample.local_translations.data()};
        api(gemx_build_skeleton(session.get(),&motion,&skeleton,error,sizeof(error)),error);
        sample.camera=translation_camera;save_pose(output_path,sample);
        std::fputs("DONE\n",stdout);std::fflush(stdout);
    }
    return 0;
}

int export_sequence(int argc,char **argv){
    if(argc!=11)throw std::invalid_argument("usage: gemx-pipeline --export DENOISER MODULE CPU|Vulkan DEVICE DESCRIPTION|- THREADS FPS MANIFEST OUTPUT");
    char error[512]{};auto cfg=config(argv[2],argv[3],argv[4],argv[6],number(argv[5]),number(argv[7]),1);
    gemx_session *raw=nullptr;api(gemx_session_create(&cfg,&raw,error,sizeof(error)),error);
    session_ptr session(raw,gemx_session_destroy);
    std::ifstream manifest(argv[9],std::ios::binary);if(!manifest)throw std::runtime_error("cannot open manifest");
    std::array<char,8> magic{};read(manifest,magic.data(),8);
    if(std::string(magic.data(),8)!="GEMMAN01")throw std::invalid_argument("wrong manifest magic");
    uint32_t count=0;read(manifest,&count);if(count<1||count>GEMX_MAX_FRAMES)throw std::invalid_argument("invalid manifest count");
    std::vector<float> positions(uint64_t(count)*77*3),rotations(uint64_t(count)*77*4),translations(uint64_t(count)*77*3);
    std::array<int32_t,77> parents{};
    for(uint32_t i=0;i<count;++i){
        uint32_t size=0;read(manifest,&size);if(!size||size>4096)throw std::invalid_argument("invalid manifest path");
        std::string path(size,'\0');read(manifest,path.data(),size);if(path.find('\0')!=std::string::npos)throw std::invalid_argument("invalid manifest path data");
        auto sample=load_pose(path);
        std::copy(sample.positions.begin(),sample.positions.end(),positions.begin()+uint64_t(i)*77*3);
        std::copy(sample.rotations.begin(),sample.rotations.end(),rotations.begin()+uint64_t(i)*77*4);
        std::copy(sample.local_translations.begin(),sample.local_translations.end(),translations.begin()+uint64_t(i)*77*3);
        if(i==0)parents=sample.parents;else if(parents!=sample.parents)throw std::invalid_argument("SOMA topology changed");
    }
    gemx_skeleton_view skeleton{count,positions.data(),rotations.data(),parents.data(),translations.data()};
    api(gemx_export_skeleton_samples_glb(session.get(),&skeleton,real(argv[8]),argv[10],error,sizeof(error)),error);
    return 0;
}
}

int main(int argc,char **argv){
    try{
        if(argc>1&&!std::strcmp(argv[1],"--worker"))return worker(argc,argv);
        if(argc>1&&!std::strcmp(argv[1],"--export"))return export_sequence(argc,argv);
        throw std::invalid_argument("expected --worker or --export");
    }catch(const std::exception &error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
