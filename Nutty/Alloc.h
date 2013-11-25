#pragma once
#include "Inc.h"

namespace nutty
{
    template <
        typename T
    >
    class CudaAllocator
    {
        typedef T* pointer;
        typedef size_t size_type;

    public:
        pointer Allocate(size_type n)
        {
            pointer ptr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMalloc((void**)&ptr, n * sizeof(T)));
            return ptr;
        }

        void Deallocate(pointer ptr)
        {
            CHECK_CNTX();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaFree((void*)ptr));
        }
    };

    template <
        typename T
    >
    class NullAllocator
    {
        typedef T* pointer;
        typedef size_t size_type;

    public:
        pointer Allocate(size_type n)
        {
            return NULL;
        }

        void Deallocate(pointer ptr)
        {

        }
    };

    template <
        typename T
    >
    class DefaultAllocator
    {
    private:
        std::allocator<T> a;
        typedef T* pointer;
        typedef size_t size_type;

    public:
        pointer Allocate(size_type n)
        {
            pointer p;
            cudaMallocHost(&p, n * sizeof(T));
            return p;
        }

        void Deallocate(pointer ptr)
        {
            cudaFreeHost(ptr);
            //free(ptr);
        }
    };
}