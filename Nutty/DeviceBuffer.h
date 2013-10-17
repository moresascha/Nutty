#pragma once
#include "base/Buffer.h"

namespace nutty
{
    template<
        typename T
    >
    class DeviceContent
    {
        typedef T* pointer;
        typedef T& type_reference;
        typedef const T& const_type_reference;
        typedef size_t size_type;

    public:

        template<
            typename C
        >
        void Insert(Iterator<T, C>& pos, const_type_reference v)
        {
            Copy(pos, v);
        }

        template<
            typename C
        >
        void Insert(Iterator<T, C>& pos, size_type d, const_type_reference v)
        {
            Copy(pos, d, v);
        }

        template<
            typename C
        >
        void Insert(Iterator<T, C>& start, Iterator<T, C>& end, const_type_reference v)
        {
            size_t d = Distance(start, end);
            Copy(start, d, v);
        }

        template<
            typename C
        >
        T operator[](Iterator<T, C>& it)
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, 1);
            return host[0];
        }
    };

    template<
        typename T
    >
    class DeviceBuffer 
        : public nutty::base::Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>>
    {

         typedef nutty::base::Base_Buffer<T, DeviceContent<T>, CudaAllocator<T>> base_class;

    public:
        DeviceBuffer(void)
            : base_class()
        {

        }

        DeviceBuffer(size_type n) 
            : base_class(n)
        {

        }

        DeviceBuffer(size_type n, const_type_reference t) 
            : base_class(n, t)
        {

        }
    };
}