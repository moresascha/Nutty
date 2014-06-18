#pragma once
#include "cuda/copy.h"
namespace nutty
{
    template <
        typename T
    >
    class Pointer
    {
    public:
        typedef T* pointer;
        typedef size_t size_type;

    private:
        pointer m_ptr;

    protected:
        __device__ __host__ Pointer(void) : m_ptr(NULL) 
        {
        }

    public:
        __device__ __host__ Pointer(const Pointer& p) : m_ptr(p.m_ptr) 
        {
        }

        __device__ __host__ Pointer(pointer ptr) : m_ptr(ptr) 
        {
        }

        __device__ __host__ pointer* GetRawPointerPtr(void) { return &m_ptr; }

        __device__ __host__ pointer operator()(void) const { return m_ptr; }

//         __device__ __host__ T operator[](size_type index)
//         {
//            return m_content[index + ptr];
//         }

        __device__ __host__ ~Pointer(void)
        {

        }

        template <
            typename T
        >
        __device__ __host__ friend Pointer<T> operator+(const Pointer<T>& i0, const Pointer<T>& i1);

        template <
            typename T
        >
        __device__ __host__ friend Pointer<T> operator-(const Pointer<T>& i0, const Pointer<T>& i1);

        template <
            typename T
        >
        __device__ __host__ friend Pointer<T> operator+(const Pointer<T>& i0, size_type i);

//         template <
//             typename T
//         >
//         __device__ __host__ friend Pointer<T> operator+(const Pointer<T>& i0, unsigned long long i);

        template <
            typename T
        >
        __device__ __host__ friend Pointer<T> operator-(const Pointer<T>& i0, size_type i);
    };

    template <
        typename T
    >
    __device__ __host__ Pointer<T> operator+(const Pointer<T>& i0, const Pointer<T>& i1)
    {
        T* p = i0.m_ptr + i1.m_ptr;
        Pointer<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ Pointer<T> operator-(const Pointer<T>& i0, Pointer<T>& i1)
    {
        T* p = i0.m_ptr - i1.m_ptr;
        Pointer<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ Pointer<T> operator+(const Pointer<T>& i0, typename Pointer<T>::size_type i)
    {
        T* p = i0.m_ptr + i;
        Pointer<T> it(p);
        return it;
    }

    template <
        typename T
    >
    __device__ __host__ Pointer<T> operator-(const Pointer<T>& i0, typename Pointer<T>::size_type i)
    {
        T* p = i0.m_ptr - i;
        Pointer<T> it(p);
        return it;
    }

//     template <
//         typename T
//     >
//     __device__ __host__ Pointer<T> operator+(Pointer<T>& i0, unsigned long long i)
//     {
//         return operator+(i0, (size_t)i);
//     }

    template <
        typename T
    >
    __device__ __host__ size_t Distance(const Pointer<T>& begin, const Pointer<T>& end)
    {
        return (end() - begin());
    }
}