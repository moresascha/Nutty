#pragma once
#include "cuda/copy.h"
namespace nutty
{
    template <
        typename T
    >
    class DevicePtr
    {
    public:
        typedef T* pointer;
        typedef size_t size_type;

    private:
        pointer m_ptr;

    public:
        __device__ __host__ DevicePtr(void) : m_ptr(NULL) { }

        __device__ __host__ DevicePtr(const DevicePtr& p) : m_ptr(p.m_ptr) { }

        __device__ __host__ DevicePtr(pointer ptr) : m_ptr(ptr) { }

        __device__ __host__ pointer* GetRawPointerPtr(void) { return &m_ptr; }

        __device__ __host__ pointer operator()(void) const { return m_ptr; }

        __device__ __host__ T operator[](size_type index)
        {
            T t;
            nutty::cuda::Copy(&t, m_ptr + index, 1, cudaMemcpyDeviceToHost);
            return t;
        }

        __device__ __host__ ~DevicePtr(void)
        {

        }

        template <
            typename T
        >
        __device__ __host__ friend DevicePtr<T> operator+(DevicePtr<T>& i0, DevicePtr<T>& i1);

        template <
            typename T
        >
        __device__ __host__ friend DevicePtr<T> operator-(DevicePtr<T>& i0, DevicePtr<T>& i1);

        template <
            typename T
        >
        __device__ __host__ friend DevicePtr<T> operator+(DevicePtr<T>& i0, size_type i);

        template <
            typename T
        >
        __device__ __host__ friend DevicePtr<T> operator+(DevicePtr<T>& i0, unsigned long long i);

        template <
            typename T
        >
        __device__ __host__ friend DevicePtr<T> operator-(DevicePtr<T>& i0, size_type i);
    };

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> operator+(DevicePtr<T>& i0, DevicePtr<T>& i1)
    {
        T* p = i0.m_ptr + i1.m_ptr;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> operator-(DevicePtr<T>& i0, DevicePtr<T>& i1)
    {
        T* p = i0.m_ptr - i1.m_ptr;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> operator+(DevicePtr<T>& i0, typename DevicePtr<T>::size_type i)
    {
        T* p = i0.m_ptr + i;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> operator-(DevicePtr<T>& i0, typename DevicePtr<T>::size_type i)
    {
        T* p = i0.m_ptr - i;
        DevicePtr<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> operator+(DevicePtr<T>& i0, unsigned long long i)
    {
        return operator+(i0, (size_t)i);
    }

    template <
        typename T
    >
    __device__ __host__ size_t Distance(const DevicePtr<T>& begin, const DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T
    >
    __device__ __host__ size_t Distance(DevicePtr<T>& begin, DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T,
        typename Iterator
    >
    __device__ __host__ size_t Distance(Iterator& begin, DevicePtr<T>& end)
    {
        return (end() - begin());
    }

    template <
        typename T,
        typename Iterator
    >
    __device__ __host__ size_t Distance(DevicePtr<T>& begin, Iterator& end)
    {
        return (end() - begin());
    }

    template <
        typename T
    >
    __device__ __host__ DevicePtr<T> DevicePtr_Cast(T* ptr)
    {
        return DevicePtr<T>(ptr);
    }
}