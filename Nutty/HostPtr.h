#pragma once
#include "Pointer.h"
#include "cuda/copy.h"
namespace nutty
{
    template <
        typename T
    >
    class HostPtr : public Pointer<T>
    {
    public:
        __device__ __host__ HostPtr(const HostPtr& p) : Pointer<T>(p)
        {
        }

        __device__ __host__ HostPtr(pointer ptr) : Pointer<T>(ptr)
        {
        }
    };

    template <
        typename T
    >
    __device__ __host__ HostPtr<T> HostPtr_Cast(void* ptr)
    {
        HostPtr<T> _ptr((T*)ptr);
        return _ptr;
    }

    template <
        typename T
    >
    __device__ __host__ const HostPtr<T> HostPtr_Cast(const void* ptr)
    {
        HostPtr<T> _ptr((T*)ptr);
        return _ptr;
    }
}