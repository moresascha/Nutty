#pragma once
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T
        >
        void Fill(        
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& begin, 
        const Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& end,
                const T& v)
        {
            Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                > it = begin;
            while(it != end)
            {
                it = v;
                it++;
            }
        }

        template <
            typename T
        >
        void Fill(        
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& begin, 
        const Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& end,
        const T& v)
        {
            //nutty::cuda::Fill(begin(), end(), v); todo
        }
    }
}