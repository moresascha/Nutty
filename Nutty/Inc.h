#pragma once
#include <Windows.h>
#include <map>
#include <string>
#include <cuda.h>
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
    void LogError(const char* format, const char* error, const char* file, int line)
    {
        std::stringstream ss;
        ss << error << " in " << file << " " << line << "\n";
        OutputDebugStringA(ss.str().c_str());
        __debugbreak();
    }
}

#ifdef _DEBUG
#define CUDA_SAFE
#endif

#ifdef CUDA_SAFE 
#define CUDA_DRIVER_SAFE_CALLING_NO_SYNC(__error__) {  \
    if (CUDA_SUCCESS != __error__) { \
    cudahErrorLog::LogError("%s\n", getCudaDrvErrorString( __error__), __FILE__, __LINE__); \
    } }
#else
#define CUDA_DRIVER_SAFE_CALLING_NO_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_DRIVER_SAFE_CALLING_SYNC(__error__) {  \
    CUresult res = __error__;\
    if (res != CUDA_SUCCESS || CUDA_SUCCESS != cuCtxSynchronize()) { \
    cudahErrorLog::LogError("%s\n", getCudaDrvErrorString( __error__), __FILE__, __LINE__); \
    } }
#else
#define CUDA_DRIVER_SAFE_CALLING_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_RT_SAFE_CALLING_NO_SYNC(__error__) {  \
    if (cudaSuccess != __error__) { \
    cudahErrorLog::LogError("%s\n", cudaGetErrorString( __error__), __FILE__, __LINE__); \
    } }
#else
#define CUDA_RT_SAFE_CALLING_NO_SYNC(__error__) __error__
#endif

#ifdef CUDA_SAFE 
#define CUDA_RT_SAFE_CALLING_SYNC(__error__) {  \
    __error__;\
    if (cudaSuccess != cudaDeviceSynchronize()) { \
    cudahErrorLog::LogError("%s\n", cudaGetErrorString( __error__), __FILE__, __LINE__); \
    } }
#else
#define CUDA_RT_SAFE_CALLING_SYNC(__error__) __error__
#endif

#define CHECK_CNTX() if(!g_CudaContext) { return; }

typedef CUdeviceptr BufferPtr;