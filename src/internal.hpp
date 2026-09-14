#pragma once
#include "gemx.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>

namespace gemx {
inline void require(bool value,const char *message){if(!value)throw std::invalid_argument(message);}
inline void write_error(char *error,uint64_t capacity,const char *message){
    if(!error || !capacity)return;
    const auto n=std::min<uint64_t>(std::strlen(message),capacity-1);
    std::memcpy(error,message,n);error[n]=0;
}
template<class F> gemx_status boundary(char *error,uint64_t capacity,F &&function) noexcept {
    if(error && capacity)error[0]=0;
    try{function();return GEMX_OK;}
    catch(const std::invalid_argument &e){write_error(error,capacity,e.what());return GEMX_INVALID_ARGUMENT;}
    catch(const std::bad_alloc &e){write_error(error,capacity,e.what());return GEMX_OUT_OF_MEMORY;}
    catch(const std::exception &e){write_error(error,capacity,e.what());return GEMX_RUNTIME_ERROR;}
    catch(...){write_error(error,capacity,"unknown error");return GEMX_RUNTIME_ERROR;}
}
inline bool finite(float value){return std::isfinite(value);}
}
