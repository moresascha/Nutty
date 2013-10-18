#pragma once
#include "../cuda/reduce.h"
#include "../Copy.h"
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& begin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& end,
        BinaryOperation op)
        {
            size_t d = Distance(begin, end);

            nutty::cuda::Reduce(begin(), begin(), d, op);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dstBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dstEnd,

        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcEnd,

        BinaryOperation op)
        {
            size_t dd = Distance(srcBegin, srcEnd);

            DeviceBuffer<T> deviceDest(dd);

            nutty::cuda::Reduce(deviceDest.Begin()(), srcBegin(), dd, op);

            size_t dh = Distance(dstBegin, dstEnd);

            nutty::Copy(dstBegin, deviceDest.Begin(), dh);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        T Reduce(
        BinaryOperation op,
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& begin, 
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& end
        )
        {
            HostBuffer<T> host(1);
            nutty::base::Reduce(host.Begin(), host.End(), begin, end, op);
            return host[0];
        }
    }
}