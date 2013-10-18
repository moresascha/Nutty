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
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dst, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& src, 
        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyHostToDevice);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dst, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& src, 
        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyDeviceToHost);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dst, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& src, 
        size_t d)
        {
            nutty::cuda::Copy(dst(), src(), d, cudaMemcpyDeviceToDevice);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dst, 
        const T& v,
        size_t d)
        {
            nutty::cuda::Copy(dst(), &v, d, cudaMemcpyHostToDevice);
        }

        template <
            typename T
        >
        void Copy(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dst, 
        const Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& src, 
        size_t d)
        {
            memcpy(dst(), src(), d);
        }
    }
}