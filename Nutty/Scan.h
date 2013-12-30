#pragma once
#include "base/scan.h"

namespace nutty
{
    template <
        typename Iterator_,
        typename ScanIterator
    >
    void Scan(Iterator_& begin, Iterator_& end, ScanIterator& scanned)
    {
        //nutty::cuda::Scan(begin(), end(), scanned(), Distance(begin, end));
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator
    >
    void Scan(Iterator_& begin, Iterator_& end, ScanIterator& scanned, SumIterator& sums)
    { 
        assert(0 && L"nyi");
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename T
    >
    void Compact(Iterator_& begin, Iterator_& end, ScanIterator& mask, ScanIterator& dstAddress, T neutral)
    {
        nutty::cuda::Compact(begin(), mask(), dstAddress(), neutral, Distance(begin, end));
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
}