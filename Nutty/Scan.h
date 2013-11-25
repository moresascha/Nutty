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
        assert(L"0");
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator
    >
    void CompactScan(Iterator_& begin, Iterator_& end, ScanIterator& scanned, SumIterator& sums)
    {
        nutty::cuda::CompactScan(begin(), end(), scanned(), sums(), Distance(begin, end));
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator
    >
    void PrefixSumScan(Iterator_& begin, Iterator_& end, ScanIterator& prefixSum, SumIterator& sums)
    {
        nutty::cuda::PrefixSumScan(begin(), end(), prefixSum(), sums(), Distance(begin, end));
    }
}