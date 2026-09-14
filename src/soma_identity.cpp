// MHR mathematical forward is adapted from the Momentum implementation in
// Meta's MHR v1.0.1 asset (MIT); SOMA transfer follows NVIDIA SOMA (Apache-2.0).
#include "soma_identity.hpp"
#include "internal.hpp"
#include <algorithm>
#include <array>
#include <cmath>

namespace gemx { namespace {
using vec=std::array<float,3>;using quat=std::array<float,4>;using mat=std::array<float,9>;
using state=std::array<float,8>;using dstate=std::array<double,8>;

vec cross(vec a,vec b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
float dot(vec a,vec b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
vec sub(vec a,vec b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
vec mul(const mat &a,vec b){return {a[0]*b[0]+a[1]*b[1]+a[2]*b[2],
    a[3]*b[0]+a[4]*b[1]+a[5]*b[2],a[6]*b[0]+a[7]*b[1]+a[8]*b[2]};}
mat mul(const mat &a,const mat &b){
    mat out{};
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)for(int k=0;k<3;++k)
        out[r*3+c]+=a[r*3+k]*b[k*3+c];
    return out;
}
mat transpose(const mat &a){return {a[0],a[3],a[6],a[1],a[4],a[7],a[2],a[5],a[8]};}
float determinant(const mat &m){return m[0]*(m[4]*m[8]-m[5]*m[7])-
    m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);}
quat qmul(quat a,quat b){return {((a[3]*b[0]+a[0]*b[3])+a[1]*b[2])-a[2]*b[1],
    ((a[3]*b[1]-a[0]*b[2])+a[1]*b[3])+a[2]*b[0],
    ((a[3]*b[2]+a[0]*b[1])-a[1]*b[0])+a[2]*b[3],
    ((a[3]*b[3]-a[0]*b[0])-a[1]*b[1])-a[2]*b[2]};}
quat normalized(quat q){
    float n=std::sqrt(dot({q[0],q[1],q[2]},{q[0],q[1],q[2]})+q[3]*q[3]);
    for(float &x:q)x/=std::max(n,1e-12f);
    return q;
}
vec rotate(quat q,vec v){q=normalized(q);vec axis{q[0],q[1],q[2]},a=cross(axis,v),aa=cross(axis,a);
    for(int k=0;k<3;++k)v[k]+=(a[k]*q[3]+aa[k])*2.f;
    return v;
}

dstate compose(dstate a,dstate b){
    auto norm=[](const dstate &s){double n=std::sqrt(s[3]*s[3]+s[4]*s[4]+s[5]*s[5]+s[6]*s[6]);
        return std::array<double,4>{s[3]/n,s[4]/n,s[5]/n,s[6]/n};};
    auto aq=norm(a),bq=norm(b);std::array<double,3> v{b[0],b[1],b[2]},axis{aq[0],aq[1],aq[2]};
    auto dcross=[](auto x,auto y){return std::array<double,3>{x[1]*y[2]-x[2]*y[1],
        x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]};};
    auto av=dcross(axis,v),aa=dcross(axis,av);dstate r{};
    for(int k=0;k<3;++k)r[k]=a[k]+a[7]*(v[k]+2.*(av[k]*aq[3]+aa[k]));
    r[3]=(aq[3]*bq[0]+aq[0]*bq[3]+aq[1]*bq[2])-aq[2]*bq[1];
    r[4]=(aq[3]*bq[1]-aq[0]*bq[2]+aq[1]*bq[3])+aq[2]*bq[0];
    r[5]=(aq[3]*bq[2]+aq[0]*bq[1]-aq[1]*bq[0])+aq[2]*bq[3];
    r[6]=(aq[3]*bq[3]-aq[0]*bq[0]-aq[1]*bq[1])-aq[2]*bq[2];r[7]=a[7]*b[7];return r;
}
state compose(state a,state b){quat aq=normalized({a[3],a[4],a[5],a[6]}),bq=normalized({b[3],b[4],b[5],b[6]});
    vec v=rotate(aq,{b[0],b[1],b[2]});state r{};for(int k=0;k<3;++k)r[k]=a[k]+a[7]*v[k];
    auto q=qmul(aq,bq);std::copy(q.begin(),q.end(),r.begin()+3);r[7]=a[7]*b[7];return r;}

mat rodrigues(vec target,vec source){
    float an=std::sqrt(dot(target,target)),bn=std::sqrt(dot(source,source));
    for(float &x:target)x/=std::max(an,1e-8f);
    for(float &x:source)x/=std::max(bn,1e-8f);
    float c=std::clamp(dot(source,target),-1.f,1.f);vec v=cross(source,target);
    if(c < -1.f+1e-6f){vec basis=std::abs(source[0])>.6f?vec{0,1,0}:vec{1,0,0};vec u=cross(source,basis);
        float n=std::sqrt(dot(u,u));for(float &x:u)x/=n;return {2*u[0]*u[0]-1,2*u[0]*u[1],2*u[0]*u[2],
            2*u[1]*u[0],2*u[1]*u[1]-1,2*u[1]*u[2],2*u[2]*u[0],2*u[2]*u[1],2*u[2]*u[2]-1};}
    mat k{0,-v[2],v[1],v[2],0,-v[0],-v[1],v[0],0},i{1,0,0,0,1,0,0,0,1};
    auto kk=mul(k,k);float f=1.f/(1.f+c);for(int n=0;n<9;++n)i[n]+=k[n]+f*kk[n];return i;
}

// One-sided Jacobi SVD is sufficient for the 3x3 covariance matrices here.
mat kabsch(const mat &h){
    mat a=h,v{1,0,0,0,1,0,0,0,1};
    for(int sweep=0;sweep<16;++sweep)for(auto pair:std::array<std::array<int,2>,3>{{{0,1},{0,2},{1,2}}}){
        int p=pair[0],q=pair[1];float app=0,aqq=0,apq=0;for(int r=0;r<3;++r){
            app+=a[r*3+p]*a[r*3+p];aqq+=a[r*3+q]*a[r*3+q];apq+=a[r*3+p]*a[r*3+q];}
        if(std::abs(apq)<=1e-7f*std::sqrt(std::max(app*aqq,0.f)))continue;
        float tau=(aqq-app)/(2*apq),t=std::copysign(1.f,tau)/(std::abs(tau)+std::sqrt(1+tau*tau));
        float c=1/std::sqrt(1+t*t),s=c*t;
        for(int r=0;r<3;++r){float x=a[r*3+p],y=a[r*3+q];a[r*3+p]=c*x-s*y;a[r*3+q]=s*x+c*y;
            x=v[r*3+p];y=v[r*3+q];v[r*3+p]=c*x-s*y;v[r*3+q]=s*x+c*y;}
    }
    std::array<int,3> order{0,1,2};std::array<float,3> sigma{};for(int c=0;c<3;++c)
        for(int r=0;r<3;++r)sigma[c]+=a[r*3+c]*a[r*3+c];
    std::sort(order.begin(),order.end(),[&](int x,int y){return sigma[x]>sigma[y];});
    mat vs{},u{};for(int c=0;c<3;++c){int source=order[c];float n=std::sqrt(sigma[source]);
        for(int r=0;r<3;++r){vs[r*3+c]=v[r*3+source];u[r*3+c]=n>1e-12f?a[r*3+source]/n:0;}}
    // Complete a rank-two basis in the same right-handed orientation.
    vec u0{u[0],u[3],u[6]},u1{u[1],u[4],u[7]},u2=cross(u0,u1);
    float n2=std::sqrt(dot(u2,u2));if(n2>1e-8f)for(float &x:u2)x/=n2;
    if(sigma[order[2]]<1e-12f){u[2]=u2[0];u[5]=u2[1];u[8]=u2[2];}
    mat result=mul(u,transpose(vs));if(determinant(result)<0){for(int r=0;r<3;++r)u[r*3+2]*=-1;result=mul(u,transpose(vs));}
    return result;
}

mat align(const std::vector<vec> &target,const std::vector<vec> &source){
    require(target.size()==source.size() && !target.empty(),"invalid SOMA rotation fit vectors");
    if(target.size()==1)return rodrigues(target[0],source[0]);
    mat h{};
    for(size_t n=0;n<target.size();++n)for(int i=0;i<3;++i)for(int j=0;j<3;++j)
        h[i*3+j]+=target[n][i]*source[n][j];
    return kabsch(h);
}
}

soma_identity_rig fit_soma_identity(const soma_identity_constants &c,const float *identity,
                                    const float *scales,float global_scale){
    require(identity && scales && std::isfinite(global_scale) && global_scale>0,
            "invalid SOMA identity input");
    require(c.mhr_offsets.size()==127*3 && c.mhr_prerotations.size()==127*4 &&
        c.mhr_parents.size()==127 && c.mhr_parameter_matrix.size()==889*204 &&
        c.mhr_inverse_bind.size()==127*8 && c.mhr_skin_weights.size()==1747 &&
        c.mhr_skin_joints.size()==1747 && c.mhr_skin_vertices.size()==1747 &&
        c.mhr_shape_vectors.size()==45*595*3 && c.mhr_base_shape.size()==595*3 &&
        c.mhr_faces.size()==1186*3 && c.transfer_face_ids.size()==4505 &&
        c.transfer_barycentric.size()==4505*4 && c.rbf_crow.size()==79 &&
        c.rbf_columns.size()==14725 && c.rbf_values.size()==14725 &&
        c.bind_world.size()==78*16 && c.soma_parents.size()==78 && c.rotation_crow.size()==79 &&
        c.rotation_vertices.size()==7709 && c.rotation_reference.size()==7709*3,
        "SOMA identity constants are unavailable");

    std::array<float,204> parameters{};std::copy_n(scales,68,parameters.begin()+136);
    std::vector<float> jp(889);for(int row=0;row<889;++row){float sum=0;const float *w=c.mhr_parameter_matrix.data()+row*204;
        for(int col=0;col<204;++col)sum+=w[col]*parameters[col];
        jp[row]=sum;
    }
    std::vector<dstate> skeleton(127);
    for(int j=0;j<127;++j){const float *p=jp.data()+j*7;
        float cy=std::cos(p[5]*.5f),sy=std::sin(p[5]*.5f),cp=std::cos(p[4]*.5f),sp=std::sin(p[4]*.5f),
              cr=std::cos(p[3]*.5f),sr=std::sin(p[3]*.5f);
        quat q{(sr*cp)*cy-(cr*sp)*sy,(cr*sp)*cy+(sr*cp)*sy,
            (cr*cp)*sy-(sr*sp)*cy,(cr*cp)*cy+(sr*sp)*sy};
        q=qmul({c.mhr_prerotations[j*4],c.mhr_prerotations[j*4+1],
                c.mhr_prerotations[j*4+2],c.mhr_prerotations[j*4+3]},q);
        dstate local{p[0]+c.mhr_offsets[j*3],p[1]+c.mhr_offsets[j*3+1],p[2]+c.mhr_offsets[j*3+2],
            q[0],q[1],q[2],q[3],std::exp(p[6]*0.69314718246459961f)};
        skeleton[j]=j?compose(skeleton[c.mhr_parents[j]],local):local;
    }
    std::vector<float> unposed(595*3);for(int v=0;v<595;++v)for(int k=0;k<3;++k){
        float value=c.mhr_base_shape[v*3+k];
        for(int n=0;n<45;++n)value+=c.mhr_shape_vectors[(n*595+v)*3+k]*identity[n];
        unposed[v*3+k]=value;
    }
    std::array<state,127> skin_state{};for(int j=0;j<127;++j){state a{},b{};
        for(int k=0;k<8;++k){a[k]=static_cast<float>(skeleton[j][k]);b[k]=c.mhr_inverse_bind[j*8+k];}
        skin_state[j]=compose(a,b);}
    std::vector<float> mhr_vertices(595*3,0);for(int i=0;i<1747;++i){int j=c.mhr_skin_joints[i],v=c.mhr_skin_vertices[i];
        auto s=skin_state[j];vec point{unposed[v*3],unposed[v*3+1],unposed[v*3+2]};
        for(float &x:point)x*=s[7];
        point=rotate({s[3],s[4],s[5],s[6]},point);
        for(int k=0;k<3;++k)mhr_vertices[v*3+k]+=(s[k]+point[k])*c.mhr_skin_weights[i];}

    std::vector<float> soma_vertices(4505*3);for(int v=0;v<4505;++v){int face=c.transfer_face_ids[v];
        int i0=c.mhr_faces[face*3],i1=c.mhr_faces[face*3+1],i2=c.mhr_faces[face*3+2];
        vec p0{mhr_vertices[i0*3],mhr_vertices[i0*3+1],mhr_vertices[i0*3+2]},
            p1{mhr_vertices[i1*3],mhr_vertices[i1*3+1],mhr_vertices[i1*3+2]},
            p2{mhr_vertices[i2*3],mhr_vertices[i2*3+1],mhr_vertices[i2*3+2]};
        vec p3=p0,normal=cross(sub(p1,p0),sub(p2,p0));for(int k=0;k<3;++k)p3[k]+=normal[k];
        const float *b=c.transfer_barycentric.data()+v*4;for(int k=0;k<3;++k)
            soma_vertices[v*3+k]=(p0[k]*b[0]+p1[k]*b[1]+p2[k]*b[2]+p3[k]*b[3])*.01f*global_scale;
    }
    soma_identity_rig out;out.fitted_positions.assign(78*3,0);out.fitted_rotations.resize(78*9);
    for(int j=0;j<78;++j){for(int i=c.rbf_crow[j];i<c.rbf_crow[j+1];++i){int v=c.rbf_columns[i];float w=c.rbf_values[i];
            for(int k=0;k<3;++k)out.fitted_positions[j*3+k]+=w*soma_vertices[v*3+k];}
        const float *bind=c.bind_world.data()+j*16;out.fitted_rotations[j*9]=bind[0];out.fitted_rotations[j*9+1]=bind[1];out.fitted_rotations[j*9+2]=bind[2];
        out.fitted_rotations[j*9+3]=bind[4];out.fitted_rotations[j*9+4]=bind[5];out.fitted_rotations[j*9+5]=bind[6];
        out.fitted_rotations[j*9+6]=bind[8];out.fitted_rotations[j*9+7]=bind[9];out.fitted_rotations[j*9+8]=bind[10];}
    // RBF intentionally leaves the non-deforming scene root at zero.
    for(int k=0;k<3;++k)out.fitted_positions[k]=c.bind_world[k*4+3];
    std::array<std::vector<int>,78> children;
    for(int j=1;j<78;++j){require(c.soma_parents[j]>=0 && c.soma_parents[j]<j,
                                  "SOMA hierarchy is not topological");children[c.soma_parents[j]].push_back(j);}
    for(int j=1;j<78;++j){
        if(children[j].empty()){
            int p=c.soma_parents[j];std::copy_n(out.fitted_rotations.begin()+p*9,9,
                                                out.fitted_rotations.begin()+j*9);continue;
        }
        std::vector<vec> target,source;target.reserve(c.rotation_crow[j+1]-c.rotation_crow[j]);
        source.reserve(target.capacity());
        for(int i=c.rotation_crow[j];i<c.rotation_crow[j+1];++i){int v=c.rotation_vertices[i];
            target.push_back({soma_vertices[v*3]-out.fitted_positions[j*3],
                soma_vertices[v*3+1]-out.fitted_positions[j*3+1],
                soma_vertices[v*3+2]-out.fitted_positions[j*3+2]});
            source.push_back({c.rotation_reference[i*3],c.rotation_reference[i*3+1],
                              c.rotation_reference[i*3+2]});
        }
        mat initial=align(target,source),bind{};std::copy_n(c.bind_world.data()+j*16,3,bind.begin());
        std::copy_n(c.bind_world.data()+j*16+4,3,bind.begin()+3);std::copy_n(c.bind_world.data()+j*16+8,3,bind.begin()+6);
        target.clear();source.clear();
        for(int child:children[j]){
            vec old{c.bind_world[child*16+3]-c.bind_world[j*16+3],
                    c.bind_world[child*16+7]-c.bind_world[j*16+7],
                    c.bind_world[child*16+11]-c.bind_world[j*16+11]};
            source.push_back(mul(initial,old));
            target.push_back({out.fitted_positions[child*3]-out.fitted_positions[j*3],
                out.fitted_positions[child*3+1]-out.fitted_positions[j*3+1],
                out.fitted_positions[child*3+2]-out.fitted_positions[j*3+2]});
        }
        mat fitted=mul(mul(align(target,source),initial),bind);
        std::copy(fitted.begin(),fitted.end(),out.fitted_rotations.begin()+j*9);
    }
    out.local_offsets.resize(78*3);
    std::copy_n(out.fitted_positions.begin(),3,out.local_offsets.begin());
    for(int j=1;j<78;++j){int p=c.soma_parents[j];vec delta{
        out.fitted_positions[j*3]-out.fitted_positions[p*3],
        out.fitted_positions[j*3+1]-out.fitted_positions[p*3+1],
        out.fitted_positions[j*3+2]-out.fitted_positions[p*3+2]};mat parent{};
        std::copy_n(out.fitted_rotations.begin()+p*9,9,parent.begin());auto local=mul(transpose(parent),delta);
        std::copy(local.begin(),local.end(),out.local_offsets.begin()+j*3);
    }
    return out;
}
}
