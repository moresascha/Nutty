#pragma once
#include "Iterator.h"

namespace nutty
{
    namespace base
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
}