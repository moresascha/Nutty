#pragma once
#include <device_launch_parameters.h>

#define NUM_BANKS 32   //fermi / kepler
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))  

namespace nutty
{
    template <
        typename T,
        T bits
    >
    __device__ __host__ __forceinline T GetMSB(T c)
    {
        T bit = bits;
        for(T i = (T)(1ULL << bits); i > 0; i>>=1)
        {
            if((c & i) == i)
            {
                return bit;
            }
            bit--;
        }
        return (T)0;
    }

    __device__ __host__ __forceinline uint GetMSB(uint c)
    {
        return GetMSB<uint, 31>(c);
    }

    __device__ __host__ __forceinline size_t GetMSB(size_t c)
    {
        return GetMSB<size_t, 63>(c);
    }

    __host__ __device__ __forceinline bool Ispow2(int c)
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

    template <
        typename T,
        class Iterator
    >
    __host__ __forceinline void ZeroMem(Iterator& begin, Iterator& end)
    {
        size_t d = Distance(begin, end);
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(begin(), 0, d * sizeof(T)));
    }

    template <
        template <class, class, class> class Buffer,
        class T,
        template <class> class Container,
        template <class> class Allocator
    >
    __host__ __forceinline void SetMem(Buffer<T, Container<T>, Allocator<T>>& b, byte value)
    {
        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemset(b.Begin()(), value, b.Size() * sizeof(T)));
    }

    namespace cuda
    {
        template <
            typename T
        >
        __host__ __device__ __forceinline T GetCudaGrid(T dataCnt, T groupSize)
        {
            if(dataCnt % groupSize == 0)
            {
                return dataCnt / groupSize;
            }
            return (dataCnt / groupSize + 1);
        }

        template <
            typename T
        >
        __host__ __device__ __forceinline T GetCudaBlock(T dataCnt, T groupSize)
        {
            return  dataCnt < groupSize ? dataCnt : groupSize;;
        }
    }
}