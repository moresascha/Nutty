#pragma once
#include "Globals.cuh"

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
        __device__ void __reduce(_SM_T s_d, _SRC_T d0, _SRC_T d1, _DST_T dst, _OPERATOR _operator)
        {
            uint id = threadIdx.x;

            s_d[id] = _operator(d0, d1);

            __syncthreads();

            for(uint i = blockDim.x/2 ; i > 0; i >>= 1)
            {
                if(id < i)
                {
                    s_d[id] = _operator(s_d[id + i], s_d[id]);
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
        __global__ void reduce(T_SRC* data, T_DST* dst, BinaryOperator _operator, uint stride, uint memoryOffset)
        {
            /*extern __shared__ T_DST s_d[];

            uint si = blockIdx.x * stride + threadIdx.x;

            T_SRC d0 = data[memoryOffset + si];
            T_SRC d1 = data[memoryOffset + si + blockDim.x];

            __reduce(s_d, d0, d1, dst, _operator);*/
        }

        /*
        template <
            typename _OPERATOR, 
            typename _SRC_T, 
            typename _DST_T
        >
        __device__ void _reduceFromIndex(_SRC_T* data, _DST_T* dst, _OPERATOR _operator, uint* index, _SRC_T extremes, uint stride, uint memoryOffset)
        {
            extern __shared__ _DST_T s_d[];

            uint si = blockIdx.x * stride + threadIdx.x;

            uint i0 = index[memoryOffset + si];
            uint i1 = index[memoryOffset + si + blockDim.x];

            _SRC_T d0;
            _SRC_T d1;

            if(i0 == INVALID_DATA_ADD)
            {
                d0 = extremes;
            }
            else
            {
                d0 = data[i0];
            }

            if(i1 == INVALID_DATA_ADD)
            {
                d1 = extremes;
            }
            else
            {
                d1 = data[i1];
            }

            __reduce(s_d, d0, d1, dst, _operator);
        } */

        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(T* dst, T* src, size_t d, BinaryOperation op)
        {
            uint elementsPerBlock = 512;

            if(elementsPerBlock > d)
            {
                elementsPerBlock = (uint)d;
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = (uint)((d / 2) / block.x);

            reduce
            <<<grid, block, block.x * 2 * sizeof(T)>>>
            (src, dst, op, elementsPerBlock, 0);

            UINT elementsLeft = (uint)d / elementsPerBlock;

            if(elementsLeft > 1)
            {
                block = elementsLeft / 2;
                grid = (uint)((d / 2) / block.x);

                reduce
                    <<<grid, block, block.x * 2 * sizeof(T)>>>
                    (dst, dst, op, elementsLeft / 2, 0);
            }
        }
    }
}