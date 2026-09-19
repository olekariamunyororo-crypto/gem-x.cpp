#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace gemx {
class model {
public:
    model(const std::string &path,ggml_backend_buffer_type_t buffer_type,
          const std::string &architecture="gemx",size_t expected_tensors=246);
    ~model();
    model(const model &)=delete;
    model &operator=(const model &)=delete;
    ggml_tensor *tensor(const std::string &name) const;
    bool contains(const std::string &name) const{return tensors_.contains(name);}
    uint32_t u32(const char *name) const;
    float f32(const char *name) const;
    std::string string(const char *name) const;
    std::vector<float> read_f32(const std::string &name) const;
    std::vector<int32_t> read_i32(const std::string &name) const;
    std::vector<std::string> string_array(const char *name) const;
private:
    gguf_context *gguf_=nullptr;
    ggml_context *context_=nullptr;
    ggml_backend_buffer_t buffer_=nullptr;
    std::unordered_map<std::string,ggml_tensor *> tensors_;
};
}
