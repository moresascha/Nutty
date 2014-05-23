#pragma once
#include "base/Buffer.h"
#include "HostBuffer.h"
#include "DevicePtr.h"

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
            Copy(pos, v, d);
        }

        template<
            typename C
        >
        void Insert(Iterator<T, C>& start, Iterator<T, C>& end, const_type_reference v)
        {
            size_t d = Distance(start, end);
            Copy(start, end, v, d);
        }

        template<
            typename C
        >
        T operator[](Iterator<const T, const C>& it) const
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, it+1);
            return host[0];
        }

        template<
            typename C
        >
        T operator[](const Iterator<T, C>& it) const
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, it+1);
            return host[0];
        }

        template<
            typename C
        >
        T operator[](const Iterator<const T, const C>& it) const
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, it+1);
            return host[0];
        }
    };

    template<
        typename T,
        typename Allocator = CudaAllocator<T>
    >
    class DeviceBuffer 
        : public nutty::base::Base_Buffer<T, DeviceContent<T>, Allocator>
    {

    public:

        typedef nutty::base::Base_Buffer<T, DeviceContent<T>, Allocator> base_class;

        DeviceBuffer(size_type n) 
            : base_class(n)
        {

        }

        DeviceBuffer(void) 
            : base_class()
        {

        }

        DeviceBuffer(pointer p, size_type n) 
            : base_class(n)
        {
            m_ptr = p;
        }

        DeviceBuffer(DeviceBuffer&& c) 
            : base_class(std::move(c))
        {

        }

        DeviceBuffer(size_type n, const_type_reference t) 
            : base_class(n, t)
        {

        }

        DeviceBuffer& operator=(DeviceBuffer&& c)
        {
            return base_class::operator=(std::move(c));
        }

        DevicePtr<T> GetDevicePtr(void)
        {
            return DevicePtr<T>(m_ptr);
        }
    };
}