#pragma once
#include "cuda_helper.h"

namespace nutty
{
    namespace cuda
    {
        template <
            typename T,
            typename Operator
        >
        __global__ void _foreach(T* data, Operator op, size_t N)
        {
            unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

            if(idx >= N)
            {
                return;
            }

            data[idx] = op(data[idx]);
        }

        template <
            typename T,
            typename Operator
        >
        void foreach(T* start, uint size, Operator op)
        {
            dim3 blockDim = cuda::GetCudaBlock(size, 256U);
            dim3 grid = cuda::GetCudaGrid(size, blockDim.x);

            _foreach<<<grid, blockDim>>>(start, op, size);
        }
    }
}