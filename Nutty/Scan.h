#pragma once
#include "base/scan.h"

namespace nutty
{
    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator,
        typename Operator
    >
    void InclusiveScan(Iterator_& begin, Iterator_& end, ScanIterator& prefixSum, SumIterator& sums, Operator op)
    {
        nutty::cuda::_InclusiveScan(begin(), prefixSum(), sums(), Distance(begin, end), op);
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator,
        typename Operator
    >
    void ExclusiveScan(Iterator_& begin, Iterator_& end, ScanIterator& scanned, SumIterator& sums, Operator op)
    { 
        nutty::cuda::_ExclusiveScan(begin(), scanned(), sums(), Distance(begin, end), op);  
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator
    >
    void ExclusivePrefixSumScan(Iterator_& begin, Iterator_& end, ScanIterator& prefixSum, SumIterator& sums)
    {
        nutty::cuda::ExclusivePrefixSumScan(begin(), prefixSum(), sums(), Distance(begin, end));
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename T
    >
    void Compact(Iterator_& dstBegin, Iterator_& begin, Iterator_& end, ScanIterator& mask, ScanIterator& dstAddress, T neutral)
    {
        nutty::cuda::Compact(dstBegin(), begin(), mask(), dstAddress(), neutral, Distance(begin, end));
    }
}