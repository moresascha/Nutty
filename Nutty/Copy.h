#pragma once
#include "base/copy.h"

namespace nutty
{
    template <
        typename IteratorOut,
        typename T
    >
    void Copy(IteratorOut& begin, const T& t, size_t d)
    {
        nutty::base::Copy(begin, t, d);
    }

    template <
        typename IteratorOut,
        typename T
    >
    void Copy(IteratorOut& begin, const T& t)
    {
        Copy(begin, t, 1);
    }

    template <
        typename IteratorIn,
        typename IteratorOut
    >
    void Copy(IteratorOut& dst, IteratorIn& src, size_t d)
    {
        nutty::base::Copy(dst, src, d);
    }

    template <
        typename IteratorIn,
        typename IteratorOut
    >
    void Copy(IteratorOut& dst, IteratorIn& srcBegin, IteratorIn& srcEnd)
    {
        nutty::base::Copy(dst, srcBegin, Distance(srcBegin, srcEnd));
    }
}