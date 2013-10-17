#pragma once
#include "../cuda/copy.h"
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T
        >
        void Copy(
                Iterator<T, Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>>
                        >& dst,
                Iterator<T, Base_Buffer<T, HostContent<T>>
                        >& src, 
                        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyHostToDevice);
        }


        template <
            typename T
        >
        void Copy(
        Iterator<T, Base_Buffer<T, HostContent<T>>
        >& dst,
        Iterator<T, Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>>
        >& src,
        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyDeviceToHost);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<T, Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>>
        >& dst,
        Iterator<T, Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>>
        >& src,
        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyDeviceToDevice);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<T, Base_Buffer<T, HostContent<T>>
        >& dst,
        Iterator<T, Base_Buffer<T, HostContent<T>>
        >& src,
        size_t d)
        {
            memcpy(dst(), src(), d);
        }
    }
}