#pragma once
#include "../cuda/reduce.h"
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

            nutty::cuda::Reduce(begin(), end(), d, op);
        }
    }
}