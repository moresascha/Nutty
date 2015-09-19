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
        T* m_pPinnedHostMemory;

    public:
        DeviceContent(void) : m_pPinnedHostMemory(NULL)
        {

        }

        ~DeviceContent(void)
        {
            if(m_pPinnedHostMemory)
            {
                cudaFreeHost(m_pPinnedHostMemory);
            }
        }

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
            //HostBuffer<T> host(1);
            //Copy(host.Begin(), it, it+1);
            if(m_pPinnedHostMemory == NULL)
            {
                cudaMallocHost((void**)&m_pPinnedHostMemory, sizeof(T));
            }
            cudaMemcpy(m_pPinnedHostMemory, it(), sizeof(T), cudaMemcpyDeviceToHost);
            return *m_pPinnedHostMemory; //host[0];
        }

        template<
            typename C
        >
        __host__ T operator[](const Iterator<T, C>& it) const
        {
            if(m_pPinnedHostMemory == NULL)
            {
                cudaMallocHost((void**)&m_pPinnedHostMemory, sizeof(T));
            }
            cudaMemcpy(m_pPinnedHostMemory, it(), sizeof(T), cudaMemcpyDeviceToHost);
            return *m_pPinnedHostMemory; //host[0];
        }

        template<
            typename C
        >
        __host__ T operator[](const Iterator<const T, const C>& it) const
        {
            //HostBuffer<T> host(1);
            //Copy(host.Begin(), it, it+1);
            if(m_pPinnedHostMemory == NULL)
            {
                cudaMallocHost((void**)&m_pPinnedHostMemory, sizeof(T));
            }
            cudaMemcpy(m_pPinnedHostMemory, it(), sizeof(T), cudaMemcpyDeviceToHost);
            return *m_pPinnedHostMemory; //host[0];
        }

        template<
            typename T
        >
        __host__ T Get(size_t index, const T* ptr) const
        {
            //HostBuffer<T> host(1);
            //Copy(host.Begin(), it, it+1);
            if(m_pPinnedHostMemory == NULL)
            {
                cudaMallocHost((void**)&m_pPinnedHostMemory, sizeof(T));
            }
            cudaMemcpy(m_pPinnedHostMemory, ptr + index, sizeof(T), cudaMemcpyDeviceToHost);
            return *m_pPinnedHostMemory; //host[0];
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

        typedef typename nutty::base::Base_Buffer<T, DeviceContent<T>, Allocator> base_class;

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

        __host__ T* GetPointer(void)
        {
            return m_ptr;
        }

        __host__ const T* __restrict GetConstPointer(void) const
        {
            return m_ptr;
        }
    };
}