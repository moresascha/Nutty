#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

namespace nutty
{
    __device__ __host__ uint getmsb(uint c)
    {
        uint bit = 31;
        for(uint i = (uint)(1 << 31); i > 0; i>>=1)
        {
            if((c & i) == i)
            {
                return bit;
            }
            bit--;
        }
        return 0;
    }

    __host__ bool ispow2(int c)
    {
        return !(c & (c-1));
    }

    namespace cuda
    {
        template <
            typename T
        >
        T getCudaGrid(T dataCnt, T groupSize)
        {
            if(dataCnt % groupSize == 0)
            {
                return dataCnt / groupSize;
            }
            return (dataCnt / groupSize + 1);
        }
    }
}