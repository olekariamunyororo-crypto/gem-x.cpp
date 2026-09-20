// Profiling-only preload: execute the real graph prefix, then repeat its first
// QKV matmul on the resident operands. Exits without returning an inference.
#include "ggml.h"
#include "ggml-backend.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>


static void dump_tensor(ggml_tensor *t,const char *name) {
    const char *dir=std::getenv("GEMX_QKV_DUMP");
    if(!dir)return;
    auto *data=std::malloc(ggml_nbytes(t));
    if(!data)std::abort();
    ggml_backend_tensor_get(t,data,0,ggml_nbytes(t));
    char path[4096];
    const int length=std::snprintf(path,sizeof path,"%s/%s",dir,name);
    if(length<0 || size_t(length)>=sizeof path)std::abort();
    FILE *f=std::fopen(path,"wb");
    if(!f || std::fwrite(data,1,ggml_nbytes(t),f)!=ggml_nbytes(t))std::abort();
    std::fclose(f);std::free(data);
}

extern "C" ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, ggml_cgraph *graph) {
    static auto real=reinterpret_cast<decltype(&ggml_backend_graph_compute_async)>(dlsym(RTLD_NEXT,"ggml_backend_graph_compute_async"));
    if(!real)std::abort();
    auto **nodes=ggml_graph_nodes(graph);
    int index=-1;
    for(int i=0;i<ggml_graph_n_nodes(graph);++i)
        if(nodes[i]->op==GGML_OP_MUL_MAT && nodes[i]->src[0] &&
           std::strcmp(nodes[i]->src[0]->name,"backbone.block.0.attn.qkv.weight")==0){index=i;break;}
    if(index<0)return real(backend,graph);
    auto *ctx=ggml_init({4*1024*1024,nullptr,true});
    if(!ctx)std::abort();
    auto *prefix=ggml_new_graph_custom(ctx,ggml_graph_n_nodes(graph)+16,false);
    for(int i=0;i<index;++i)ggml_graph_add_node(prefix,nodes[i]);
    if(real(backend,prefix)!=GGML_STATUS_SUCCESS)std::abort();
    ggml_backend_synchronize(backend);
    auto *single=ggml_new_graph_custom(ctx,64,false);
    for(int i=0;i<32;++i)ggml_graph_add_node(single,nodes[index]);
    auto *qkv=nodes[index];
    std::fprintf(stderr,"isolated QKV: M=%lld N=%lld K=%lld batch=%lld; real first-block input, resident warm-cache replay\n",
        (long long)qkv->ne[0],(long long)qkv->ne[1],(long long)qkv->src[0]->ne[0],(long long)(qkv->ne[2]*qkv->ne[3]));
    dump_tensor(qkv->src[0],"weights.f32");
    dump_tensor(qkv->src[1],"input.f32");
    const auto start=std::chrono::steady_clock::now();
    unsigned count=0;
    do {
        if(real(backend,single)!=GGML_STATUS_SUCCESS)std::abort();
        ggml_backend_synchronize(backend);count+=32;
    } while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<15);
    const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::fprintf(stderr,"isolated QKV: iterations=%u host-inclusive-ms=%.6f\n",count,ms/count);
    dump_tensor(qkv,"output.f32");
    std::fflush(nullptr);std::_Exit(0);
}
