#pragma once
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

#define NUM_BANKS 32   //fermi / kepler
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

namespace nutty
{
    __device__ __host__ __forceinline uint GetMSB(uint c)
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

    __host__ __forceinline bool Ispow2(int c)
    {
        return !(c & (c-1));
    }

    template <
        template <class> class Pointer, class T
    >
    __host__ __forceinline void ZeroMem(Pointer<T>& b)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(b.Begin()(), 0, b.Size() * sizeof(T)));
    }

    template <
        template <class, class, class> class Buffer,
        class T,
        template <class> class Container,
        template <class> class Allocator
    >
    __host__ __forceinline void ZeroMem(Buffer<T, Container<T>, Allocator<T>>& b)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(b.Begin()(), 0, b.Size() * sizeof(T)));
    }

    namespace cuda
    {
        template <
            typename T
        >
        __forceinline T GetCudaGrid(T dataCnt, T groupSize)
        {
            if(dataCnt % groupSize == 0)
            {
                return dataCnt / groupSize;
            }
            return (dataCnt / groupSize + 1);
        }
    }
}