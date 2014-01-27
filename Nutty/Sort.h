#pragma once
#include "base/sort.h"

namespace nutty
{
    template <
        typename IteratorKey,
        typename DataIterator,
        typename BinaryOperation
    >
    __device__ __host__ void Sort(IteratorKey& keyBegin, IteratorKey& keyEnd, DataIterator& values, BinaryOperation op)
    {
        nutty::base::Sort(keyBegin, keyEnd, values, op);
    }

    template <
        typename DataIterator,
        typename BinaryOperation
    >
    __device__ __host__ void Sort(DataIterator& begin, DataIterator& end, BinaryOperation op)
    {
        nutty::base::Sort(begin, end, op);
    }

    template <
        typename DataIterator,
        typename BinaryOperation
    >
    __device__ __host__ void Sort(DataIterator& begin, size_t d, BinaryOperation op)
    {
        nutty::base::Sort(begin, d, op);
    }
}