#pragma once
#include "Cudah.h"

namespace cudahu
{
    nutty::nutty* Init(LPCSTR projectDir, ID3D11Device* device = NULL);

    VOID Destroy(VOID);

    template <typename T>
    VOID PrintArray(T* t, UINT l, UINT stride = -1)
    {
        for(UINT i = 0; i < l; ++i)
        {
            if(i != 0 && (i % stride == 0))
            {
                DEBUG_OUT("\n");
            }
            DEBUG_OUT_A("%u ", t[i]);
        }
        DEBUG_OUT("\n\n");
    }

    VOID PrintArrayF(FLOAT* t, UINT l, UINT stride = -1);

    template<typename T>
    VOID reduce43CPU(float4* data, UINT l, float3& result, T t, UINT offset)
    {
        for(UINT i = 0; i < l; ++i)
        {
            result = t(*((float3*)(data+offset+i)), result);
        }
    }

    VOID reduce43IndexedGPU(
        nutty::cuBuffer data, 
        nutty::cuBuffer index, 
        nutty::cuda_kernel kernel,
        UINT gridDim, 
        UINT blockDim,
        nutty::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduce43MinIndexedGPU(
        nutty::cuBuffer data, 
        nutty::cuBuffer index, 
        UINT gridDim, 
        UINT blockDim,
        nutty::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduce43MaxIndexedGPU(
        nutty::cuBuffer data, 
        nutty::cuBuffer index, 
        UINT gridDim, 
        UINT blockDim,
        nutty::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduceMin43GPU(
        nutty::cuBuffer data, 
        UINT count, 
        UINT elementsPerBlock,
        nutty::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduceMax43GPU(
        nutty::cuBuffer data, 
        UINT count, UINT elementsPerBlock,
        nutty::cuBuffer result,
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduceMin43GPU1S(
        nutty::cuBuffer data,
        UINT count, UINT elementsPerBlock,
        nutty::cuBuffer result, nutty::nutty* c,
        UINT stride,
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID reduceMax43GPU1S(
        nutty::cuBuffer data,
        UINT count, UINT elementsPerBlock,
        nutty::cuBuffer result, nutty::nutty* c,
        UINT stride,
        UINT memoryOffset, 
        float3* cpu_result = NULL);

    VOID bitonicSortFloatPerGroup(nutty::cuBuffer data, VOID* cpu_result = NULL);

    VOID bitonicSortFloatPerGroup(nutty::cuBuffer data, UINT startStage, UINT elementsPerBlock, VOID* cpu_result = NULL);

    VOID bitonicSortFloat(nutty::cuBuffer data, VOID* cpu_result = NULL);

    VOID bitonicSortFloatFlat(nutty::cuBuffer data, VOID* cpu_result = NULL);
}