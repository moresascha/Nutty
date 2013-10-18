#pragma once
#include "base/copy.h"

namespace nutty
{
    template <
        typename T, 
        typename C
    >
    void Copy(Iterator<T, C>& dstBegin, Iterator<T, C>& dstEnd, Iterator<T, C>& srcBegin, Iterator<T,C>& srcEnd)
    {

    }

    template <
        typename T, 
        typename C
    >
    void Copy(Iterator<T, C>& begin, const T& t, size_t d)
    {
        nutty::base::Copy(begin, t, d);
    }

    template <
        typename T, 
        typename C
    >
    void Copy(Iterator<T, C>& begin, const T& t)
    {
        nutty::base::Copy(begin, t, 1);
    }

    template <
        typename T, 
        typename C,
        typename D
    >
    void Copy(Iterator<T, C>& dst, Iterator<T, D>& src, size_t d)
    {
        nutty::base::Copy(dst, src, d);
    }
}