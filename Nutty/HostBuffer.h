#pragma once
#include "base/Buffer.h"

namespace nutty
{
    template <
        typename T
    >
    class HostContent
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
            *(pos()) = v;
        }

        template<
            typename C
        >
        void Insert(Iterator<T, C>& start, Iterator<T, C>& end, const_type_reference v)
        {
            //nutty::base::Copy(start, end, v);
        }

        template<
            typename C
        >
        void Insert(Iterator<T, C>& pos, size_t d, const_type_reference v)
        {
            //nutty::base::Copy(pos, d, v);
        }

        template<
            typename C
        >
        T operator[](Iterator<T, C>& index)
        {
            return *(index());
        }
    };

    template <
        typename T
    >
    class HostBuffer 
        : public nutty::base::Base_Buffer<T, HostContent<T>>
    {

        typedef nutty::base::Base_Buffer<T, HostContent<T>> base_class;

    public:
        HostBuffer(size_type n) 
            : base_class(n)
        {

        }

        HostBuffer(void)
        {

        }

        HostBuffer(size_type n, const T& t) 
            : base_class(n, t)
        {

        }

        T& operator[](size_type index)
        {
            return *(m_ptr + index);
        }
    };
}