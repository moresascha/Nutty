#pragma once
#include "Inc.h"
#include "base/foreach.h"

namespace nutty
{
    template <
        typename Iterator_,
        typename Operation
    >
    void ForEach(Iterator_& start, Iterator_& end, Operation op)
    {
        base::ForEach(start, end, op);
    }
}