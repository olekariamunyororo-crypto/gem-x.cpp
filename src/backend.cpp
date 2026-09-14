#include "backend.hpp"
#include "internal.hpp"
#include <filesystem>
#include <map>
#include <mutex>

namespace gemx { namespace {
std::mutex registry_mutex;
std::map<std::string,ggml_backend_reg_t> registries;
}

backend::backend(const std::string &module,const std::string &name,uint32_t index,
                 uint32_t threads,const std::string &expected_description){
    require(name=="CPU" || name=="Vulkan","backend must be CPU or Vulkan");
    require(threads>=1 && threads<=1024,"invalid backend thread count");
    require(std::filesystem::is_regular_file(module) || std::filesystem::is_directory(module),
            "backend module or variant directory does not exist");
    const auto canonical=std::filesystem::canonical(module).string();
    ggml_backend_reg_t registry=nullptr;
    {
        std::lock_guard lock(registry_mutex);
        auto found=registries.find(canonical);
        if(found!=registries.end())registry=found->second;
        else if(std::filesystem::is_directory(canonical)){
            ggml_backend_load_all_from_path(canonical.c_str());
            registry=ggml_backend_reg_by_name(name.c_str());
        }else registry=ggml_backend_load(canonical.c_str());
        if(registry)registries.emplace(canonical,registry);
    }
    require(registry!=nullptr,"cannot load backend module");
    require(name==ggml_backend_reg_name(registry),"backend module has unexpected registry name");
    require(index<ggml_backend_reg_dev_count(registry),"backend device index is unavailable");
    auto device=ggml_backend_reg_dev_get(registry,index);
    require(expected_description.empty() || expected_description==ggml_backend_dev_description(device),
            "backend device description does not match");
    handle_=ggml_backend_dev_init(device,nullptr);
    if(!handle_)throw std::runtime_error("backend initialization failed");
    try{
        auto setter=reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(registry,"ggml_backend_set_n_threads"));
        if(name=="CPU")require(setter!=nullptr,"CPU backend lacks thread control");
        if(setter)setter(handle_,static_cast<int>(threads));
        description_=std::string(ggml_backend_dev_name(device))+": "+ggml_backend_dev_description(device);
    }catch(...){
        ggml_backend_free(handle_);
        handle_=nullptr;
        throw;
    }
}

backend::~backend(){
    if(handle_)ggml_backend_free(handle_);
}

ggml_backend_buffer_type_t backend::buffer_type() const{
    return ggml_backend_get_default_buffer_type(handle_);
}

void backend::compute(ggml_cgraph *graph){
    for(int i=0;i<ggml_graph_n_nodes(graph);++i){
        auto *node=ggml_graph_node(graph,i);
        if(!ggml_backend_supports_op(handle_,node))
            throw std::runtime_error(std::string("backend cannot execute ")+ggml_op_name(node->op));
    }
    if(ggml_backend_graph_compute(handle_,graph)!=GGML_STATUS_SUCCESS)
        throw std::runtime_error("backend graph computation failed");
}
}
