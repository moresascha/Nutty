#pragma once
#include "cuda/copy.h"
//#include "DeviceBuffer.h"
#include "Pointer.h"

namespace nutty
{
    template <
        typename T
    >
    class DevicePtr : public nutty::Pointer <
        T
    >
    {
        typedef nutty::Pointer<T> base_class;
    public:
        __device__ __host__ DevicePtr(const DevicePtr& p) 
            : base_class(p)
        {
        }

        __device__ __host__ DevicePtr(pointer ptr) 
            : base_class(ptr)
        {
        }

        __device__ __host__ DevicePtr(void) 
            : base_class(NULL)
        {
        }
    };

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> DevicePtr_Cast(void* ptr)
    {
        DevicePtr<T> _ptr((T*)ptr);
        return _ptr;
    }
}