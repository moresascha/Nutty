#pragma once
#include "Iterator.h"
#include "cuda/foreach.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T,
            typename Operation
        >
        void ForEach(
        Iterator<T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
        >& start, 
        Iterator<T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
        >& end, 
        Operation op)
        {
            for(Iterator<T, HostContent<T>> s = start; s != end; ++s)
            {
                op(*s);
            }
        }

        template <
            typename T,
            typename Operation
        >
        void ForEach(
        Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& start, 
        Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& end, 
        Operation op)
        {
            size_t size = nutty::Distance(start, end);
            nutty::cuda::foreach(start(), (uint)size, op);
        }
    }
}