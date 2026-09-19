// Optional Linux LD_PRELOAD trace. CPU API durations, not GPU kernel timings.
// Use GEMX_GGML_TRACE=/path/trace.csv; never preload into the production demo.
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>

namespace {
double now(clockid_t clock){
    timespec t{};clock_gettime(clock,&t);return t.tv_sec+t.tv_nsec*1e-9;
}
struct trace {
    const char *name;size_t bytes;
    double start=now(CLOCK_MONOTONIC),cpu=now(CLOCK_THREAD_CPUTIME_ID);
    ~trace(){
        const double end=now(CLOCK_MONOTONIC),cpu_end=now(CLOCK_THREAD_CPUTIME_ID);
        static FILE *file=[](){
            const char *path=std::getenv("GEMX_GGML_TRACE");
            FILE *f=path?std::fopen(path,"w"):nullptr;
            if(path&&!f){std::perror("GEMX_GGML_TRACE");std::abort();}
            if(f)std::fprintf(f,"operation,start_seconds,end_seconds,wall_ms,cpu_ms,bytes\n");
            return f;
        }();
        if(file)std::fprintf(file,"%s,%.9f,%.9f,%.6f,%.6f,%zu\n",name,start,end,
            (end-start)*1000,(cpu_end-cpu)*1000,bytes);
    }
};
template<class T>T resolve(const char *name){
    auto fn=reinterpret_cast<T>(dlsym(RTLD_NEXT,name));
    if(!fn){std::fprintf(stderr,"cannot resolve %s: %s\n",name,dlerror());std::abort();}
    return fn;
}
}
extern "C" {
void ggml_backend_tensor_set(ggml_tensor *tensor,const void *data,size_t offset,size_t size){
    static auto real=resolve<decltype(&ggml_backend_tensor_set)>("ggml_backend_tensor_set");
    trace t{"upload",size};real(tensor,data,offset,size);
}
void ggml_backend_tensor_get(const ggml_tensor *tensor,void *data,size_t offset,size_t size){
    static auto real=resolve<decltype(&ggml_backend_tensor_get)>("ggml_backend_tensor_get");
    trace t{"download",size};real(tensor,data,offset,size);
}
ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend,ggml_cgraph *graph){
    static auto real=resolve<decltype(&ggml_backend_graph_compute_async)>("ggml_backend_graph_compute_async");
    trace t{"record_submit",0};return real(backend,graph);
}
void ggml_backend_synchronize(ggml_backend_t backend){
    static auto real=resolve<decltype(&ggml_backend_synchronize)>("ggml_backend_synchronize");
    trace t{"synchronize",0};real(backend);
}
}
