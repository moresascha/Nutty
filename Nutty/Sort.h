#pragma once

#include "base/sort.h"

namespace nutty
{
    template <
        typename IteratorKey,
        typename DataIterator,
        typename BinaryOperation
    >
    ___host void Sort(IteratorKey& keyBegin, IteratorKey& keyEnd, DataIterator& values, BinaryOperation op, cudaStream_t pStream = NULL)
    {
        nutty::base::Sort(keyBegin, keyEnd, values, op, pStream);
    }

    template <
        typename DataIterator,
        typename BinaryOperation
    >
    ___host void Sort(DataIterator& begin, DataIterator& end, BinaryOperation op, cudaStream_t pStream = NULL)
    {
        nutty::base::Sort(begin, end, op, pStream);
    }

    template <
        typename DataIterator,
        typename BinaryOperation
    >
    ___host void Sort(DataIterator& begin, size_t d, BinaryOperation op, cudaStream_t pStream = NULL)
    {
        nutty::base::Sort(begin, d, op, pStream);
    }
}