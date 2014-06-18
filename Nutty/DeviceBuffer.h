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
        __host__ void Insert(Iterator<T, C>& pos, const_type_reference v)
        {
            Copy(pos, v);
        }

        template<
            typename C
        >
        __host__ void Insert(Iterator<T, C>& pos, size_type d, const_type_reference v)
        {
            Copy(pos, v, d);
        }

        template<
            typename C
        >
        __host__ void Insert(Iterator<T, C>& start, Iterator<T, C>& end, const_type_reference v)
        {
            size_t d = Distance(start, end);
            Copy(start, end, v, d);
        }

        template<
            typename C
        >
        __host__ T operator[](Iterator<const T, const C>& it) const
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, it+1);
            return host[0];
        }

        template<
            typename C
        >
        __host__ T operator[](const Iterator<T, C>& it) const
        {
            HostBuffer<T> host(1);
            Copy(host.Begin(), it, it+1);
            return host[0];
        }

        template<
            typename C
        >
        __host__ T operator[](const Iterator<const T, const C>& it) const
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

        __host__ DeviceBuffer(size_type n) 
            : base_class(n)
        {

        }

        __host__ DeviceBuffer(void) 
            : base_class()
        {

        }

        __host__ DeviceBuffer(pointer p, size_type n) 
            : base_class(n)
        {
            m_ptr = p;
        }

        __host__ DeviceBuffer(DeviceBuffer&& c) 
            : base_class(std::move(c))
        {

        }

        __host__ DeviceBuffer(size_type n, const_type_reference t) 
            : base_class(n, t)
        {

        }

        __host__ DeviceBuffer& operator=(DeviceBuffer&& c)
        {
            return base_class::operator=(std::move(c));
        }

        __host__ DevicePtr<T> GetDevicePtr(void)
        {
            return DevicePtr<T>(m_ptr);
        }
    };
}