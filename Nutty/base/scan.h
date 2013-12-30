#pragma once
#include "../DeviceBuffer.h"
#include "../DevicePtr.h"
#include "../cuda/scan.h"
#include "Iterator.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T,
            typename C
        >
        void Scan(        
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& begin,
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& end,
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& scanned)
        {
            nutty::cuda::ExclusiveScan(begin(), end(), scanned(), Distance(begin, end));
        }

        template <
            typename T
        >
        void Scan(DevicePtr<T> begin, DevicePtr<T> end, DevicePtr<T> scanned)
        {
            nutty::cuda::ExclusiveScan(begin(), end(), scanned(), Distance(begin, end));
        }
    }
}