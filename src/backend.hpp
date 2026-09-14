#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include <cstdint>
#include <string>

namespace gemx {
class backend {
public:
    backend(const std::string &module,const std::string &name,uint32_t index,
            uint32_t threads,const std::string &expected_description);
    ~backend();
    backend(const backend &)=delete;
    backend &operator=(const backend &)=delete;
    ggml_backend_t handle() const{return handle_;}
    ggml_backend_buffer_type_t buffer_type() const;
    const std::string &description() const{return description_;}
    void compute(ggml_cgraph *graph);
private:
    ggml_backend_t handle_=nullptr;
    std::string description_;
};
}
