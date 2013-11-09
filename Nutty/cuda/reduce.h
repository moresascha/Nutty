#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"

namespace nutty
{
    namespace cuda
    {
        template <
            typename _SM_T, 
            typename _SRC_T, 
            typename _DST_T, 
            typename _OPERATOR
        >
        __device__ void __reduce(_SM_T s_d, _SRC_T d0, _SRC_T d1, _DST_T dst, uint startStage, _OPERATOR _operator)
        {
            uint id = threadIdx.x;

            s_d[id] = _operator(d0, d1);

            __syncthreads();

            for(int i = startStage; i > 0; i >>= 1)
            {
                if(id < i)
                {
                    if(id + i < blockDim.x)
                    {
                        s_d[id] = _operator(s_d[id + i], s_d[id]);
                    }
                }
                
                __syncthreads();
            }
            if(id == 0)
            {
                dst[blockIdx.x] = s_d[0];
            } 
        }

        template< 
            typename T_SRC, 
            typename T_DST,
            typename BinaryOperator
        >
        __global__ void reduce(T_SRC* data, T_DST* dst, BinaryOperator _operator, uint stride, uint startStage, uint length, uint memoryOffset)
        {
            uint si = blockIdx.x * stride + threadIdx.x;

            T_SRC d0 = data[memoryOffset + si];
            T_SRC d1 = d0;

            if(memoryOffset + si + blockDim.x < length)
            {
                d1 = data[memoryOffset + si + blockDim.x];
            }

            ShrdMemory<T_DST> shrd;

            __reduce(shrd.Ptr(), d0, d1, dst, startStage, _operator);
        }

        template< 
            typename T_SRC, 
            typename T_DST,
            typename T_ID,
            typename BinaryOperator
        >
        __global__ void reduceIndexed(T_SRC* data, T_DST* dst, BinaryOperator _operator, T_ID* index, 
                                      T_SRC extremes, uint invalidIndex, uint stride, uint startStage, uint length, uint memoryOffset)
        {
            ShrdMemory<T_DST> shrd;

            uint si = blockIdx.x * stride + threadIdx.x;

            T_ID i0 = index[memoryOffset + si];
            T_ID i1 = i0;

            if(memoryOffset + si + blockDim.x < length)
            {
                i1 = index[memoryOffset + si + blockDim.x];
            }

            T_SRC d0;
            T_SRC d1;

            if(i0 == invalidIndex)
            {
                d0 = extremes;
            }
            else
            {
                d0 = data[i0];
            }

            if(i1 == invalidIndex)
            {
                d1 = extremes;
            }
            else
            {
                d1 = data[i1];
            }

            __reduce(shrd.Ptr(), d0, d1, dst, startStage, _operator);
        }

        template <
            typename T,
            typename T_ID,
            typename BinaryOperation
        >
        void ReduceIndexed(T* dst, T* src, size_t d, T_ID* index, T_ID invalidIndex, T extreme, BinaryOperation op, uint elementsPerBlock, uint memoryOffset = 0)
        {
            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d%2);
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = max(1, (uint)((d / 2) / block.x));

            uint startStage = elementsPerBlock;

            if(!ispow2(startStage))
            {
                startStage = 1 << (getmsb(d)+1);
            }

            reduceIndexed
                <<<grid, block, block.x * sizeof(T)>>>
                (src, dst, op, index, extreme, invalidIndex, elementsPerBlock, startStage, memoryOffset + (uint)d, memoryOffset);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(T* dst, T* src, size_t d, BinaryOperation op, uint elementsPerBlock, uint memoryOffset = 0)
        {
            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d%2);
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = max(1, (uint)((d / 2) / block.x));

            uint startStage = elementsPerBlock;

            if(!ispow2(startStage))
            {
                startStage = 1 << (getmsb(d)+1);
            }

            reduce
                <<<grid, block, block.x * sizeof(T)>>>
                (src, dst, op, elementsPerBlock, startStage, memoryOffset + d, memoryOffset);
        }
    }
}