#pragma once
#include "base/copy.h"

namespace nutty
{
    template <
        typename T, 
        typename C
    >
    void Copy(Iterator<T, C>& srcStart, Iterator<T,C>& srcEnd, Iterator<T, C>& dstStart, Iterator<T, C>& dstEnd)
    {

    }

    template <
        typename T, 
        typename C
    >
    void Copy(Iterator<T, C>& dst, Iterator<T, C>& src, size_t d)
    {
        nutty::base::Copy(dst, src, d);
    }
}