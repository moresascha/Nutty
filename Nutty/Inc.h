#pragma once
#include <Windows.h>
#include <map>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <vector_types.h>
#include "inc/drvapi_error_string.h"
#include <algorithm>
#include <assert.h>
#include "Helper.h"
#include <sstream>

typedef unsigned int uint;

namespace cudahErrorLog
{
    __host__ void __forceinline LogError(const char* format, const char* error, const char* file, int line, int errorEnum)
    {
        std::stringstream ss;
        ss << (error?error:"error==NULL") << " in " << file << " line=" << line << " (enum=)" << errorEnum << "\n";
        OutputDebugStringA(ss.str().c_str());
       __debugbreak();
    }
}

#ifdef NUTTY_DEBUG
    #define CUDA_SAFE
#endif

#ifdef CUDA_SAFE
    #define DEVICE_SYNC_CHECK() CUDA_RT_SAFE_CALLING_NO_SYNC(cudaDeviceSynchronize())
#else
    #define DEVICE_SYNC_CHECK()
#endif

#ifdef CUDA_SAFE 
#define CUDA_DRIVER_SAFE_CALLING_NO_SYNC(__error__) {  \
    CUresult __res = __error__; \
    if (CUDA_SUCCESS != __res) { \
    cudahErrorLog::LogError("%s\n", getCudaDrvErrorString(__res), __FILE__, __LINE__, __res); \
    } }
#else
#define CUDA_DRIVER_SAFE_CALLING_NO_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_DRIVER_SAFE_CALLING_SYNC(__error__) {  \
    CUresult __res = __error__;\
    if (__res != CUDA_SUCCESS || CUDA_SUCCESS != cuCtxSynchronize()) { \
    cudahErrorLog::LogError("%s\n", getCudaDrvErrorString(__error__), __FILE__, __LINE__, __res); \
    } }
#else
#define CUDA_DRIVER_SAFE_CALLING_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_RT_SAFE_CALLING_NO_SYNC(__error__) {  \
    cudaError_t __res = __error__; \
    if (cudaSuccess != __res) { \
    cudahErrorLog::LogError("%s\n", cudaGetErrorString(__res), __FILE__, __LINE__, __res); \
    } }
#else
#define CUDA_RT_SAFE_CALLING_NO_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_RT_SAFE_CALLING_SYNC(__error__) {  \
    cudaError_t __res = __error__; \
    __res = cudaDeviceSynchronize(); \
    if (cudaSuccess != cudaDeviceSynchronize()) { \
    cudahErrorLog::LogError("%s\n", cudaGetErrorString(__res), __FILE__, __LINE__, __res); \
    } }
#else
#define CUDA_RT_SAFE_CALLING_SYNC(__error__) __error__
#endif

#define CHECK_CNTX() if(!g_CudaContext) { return; }

typedef CUdeviceptr BufferPtr;