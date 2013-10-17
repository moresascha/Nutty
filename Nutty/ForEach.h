#pragma once

#include "Inc.h"

namespace nutty
{
    template <
        typename T, 
        typename C,
        typename Operation
    >
    void ForEach(Iterator<T, C>& start, Iterator<T, C>& end, Operation op)
    {
        for(Iterator<T, C> s = start; s != end; ++s)
        {
            op(*s);
        }
    }
}