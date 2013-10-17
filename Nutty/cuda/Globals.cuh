#pragma once
#include <device_launch_parameters.h>
#include "../inc/cutil_math.h"

#define INVALID_DATA_ADD ((uint)-1)
#define FLT_MAX 3.402823466e+38F

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

template<typename T>
struct ShrdMemory
{
    void error();

    T* Ptr(void) 
    { 
        error(); 
        return 0;
    }
};

template<>
struct ShrdMemory<int>
{
    __device__ int* Ptr(void) 
    { 
        extern __device__ __shared__ int s_int[];
        return s_int;
    }
};

template<>
struct ShrdMemory<uint>
{
    __device__ uint* Ptr(void) 
    { 
        extern __device__ __shared__ uint s_uint[];
        return s_uint;
    }
};

template<>
struct ShrdMemory<float>
{
    __device__ float* Ptr(void) 
    { 
        extern __device__ __shared__ float s_float[];
        return s_float;
    }
};

template<>
struct ShrdMemory<float3>
{
    __device__ float3* Ptr(void) 
    { 
        extern __device__ __shared__ float3 s_float3[];
        return s_float3;
    }
};