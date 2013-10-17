#pragma once
#include "Globals.cuh"

namespace nutty
{
    namespace cuda
    {
        template<
            typename T, 
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortStep(T* v, uint stage, uint step, uint id, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (id / step);
            uint second = first + step;

            uint bankOffset = (id % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            T n0 = v[offset + first];

            T n1 = v[offset + second];

            char dir = (((2 * GlobalId ) / stage) & 1);  //order & ((((2 * id ) / stage) & 1) + 1);
            char cmp = _cmp_func(n0, n1);

            if((!dir & cmp) | (dir & !cmp))
            {
                v[offset + first] =  n1;
                v[offset + second] =  n0;
            }
        }

        template<
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortStep(T* v, uint stage, uint step, BinaryOperation _cmp_func, uint offset = 0)
        {
            __bitonicMergeSortStep(v, stage, step, GlobalId, _cmp_func, offset);
        }

        template<
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortPerGroup(T* g_values, uint startStage, uint startStep, uint length, BinaryOperation _cmp_func)
        {
            //char order = blockIdx.x % 2 + 1;
            uint tId = threadIdx.x;
            uint elementsPerBlock = 2 * blockDim.x;
            uint i = blockIdx.x * elementsPerBlock + 2 * tId;

            ShrdMemory<T> shrdMem;
            T* shrd = shrdMem.Ptr();

            shrd[2 * tId + 0] = g_values[i + 0];
            shrd[2 * tId + 1] = g_values[i + 1];

            uint step = startStep;

            for(uint stage = startStage; stage <= length;)
            {
                for(;step > 0; step = step >> 1)
                {
                    __syncthreads();
                    __bitonicMergeSortStep(shrd, stage, step, tId, _cmp_func);
                }
                stage <<= 1;
                step = stage >> 1;
            }

            g_values[i + 0] = shrd[2 * tId + 0];
            g_values[i + 1] = shrd[2 * tId + 1];
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void SortPerGroup(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& start, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>> 
                >& end, 
        uint elementsPerBlock, uint startStage, uint startStep, uint length, BinaryOperation op)
        {
            uint d = (uint)Distance(start, end);
            dim3 block = elementsPerBlock / 2; 
            dim3 grid = (d / 2) / block.x;

            uint shrdMem = elementsPerBlock * sizeof(T);

            bitonicMergeSortPerGroup
            <<<grid, block, shrdMem>>>
            (
            start(), startStage, startStep, length, op
            );
        }

        template<
            typename T, 
            typename BinaryOperation
        >
        void SortStep(
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& start, 
        uint grid, uint block, uint stage, uint step, BinaryOperation op, uint offset = 0)
        {
            bitonicMergeSortStep<            
                T, 
                BinaryOperation
            >
            <<<grid, block>>>
            (
            start(), stage, step, op
            );
        }
    }
}