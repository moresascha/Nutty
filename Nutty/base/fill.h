#pragma once
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    namespace base
    {
        template <
            typename T,
            typename Generator
        >
        void Fill(        
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& begin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& end, Generator g)
        {
            for(auto s = begin; s != end; ++s)
            {
                s = g();
            }
        }

        template <
            typename T,
            typename Generator
        >
        void Fill(        
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& begin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& end, Generator g)
        {
            size_t d = Distance(begin, end);

            nutty::HostBuffer<T> b(d);

            for(auto s = b.Begin(); s != b.End(); ++s)
            {
                s = g();
            }

            Copy(begin, b.Begin(), d);
        }

        template <
            typename T
        >
        void Fill(        
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& begin, 
        Iterator<
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
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& end,
        const T& v)
        {
            size_t d = Distance(begin, end);
            nutty::HostBuffer<T> host(d, v);
            Copy(begin, host.Begin(), d);
        }
    }
}