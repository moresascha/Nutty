#pragma once
#include <device_launch_parameters.h>
#include "../inc/cutil_math.h"

#define INVALID_DATA_ADD ((uint)-1)
#define FLT_MAX 3.402823466e+38F

#define GlobalId (blockDim.x * blockIdx.x + threadIdx.x)

template <
    typename K,
    typename V
    >
struct KVPair
{
    __device__ KVPair(K _k, V _v) : k(_k), v(_v)
    {

    }

    K k;
    V v;
};

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

template< template <typename, typename> class KVPair, class K, class V>
struct ShrdMemory<KVPair<K, V>>
{
    __device__ KVPair<K, V>* Ptr(void) 
    { 
        extern __device__ __shared__ KVPair<K, V> s_kv[];
        return s_kv;
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

template<>
struct ShrdMemory<float2>
{
    __device__ float2* Ptr(void) 
    { 
        extern __device__ __shared__ float2 s_float2[];
        return s_float2;
    }
};
