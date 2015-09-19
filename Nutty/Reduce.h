#pragma once
#include "base/reduce.h"

namespace nutty
{
    template <
        typename T,
        typename T_ID,
        typename C,
        typename C1,
        typename A,
        typename A1,
        typename BinaryOperation
    >
    void ReduceIndexed(base::Base_Buffer<T, C, A>& dst, base::Base_Buffer<T, C, A>& src, base::Base_Buffer<T_ID, C1, A1>& index, T& extreme, BinaryOperation op)
    {
        nutty::base::ReduceIndexed(dst.Begin(), src.Begin(), src.End(), index.Begin(), extreme, op);
    }

    template <
        typename IteratorDst,
        typename IteratorSrc,
        typename BinaryOperation,
        typename T
    >
    __host__ void Reduce(IteratorDst& dst, IteratorSrc& srcBegin, IteratorSrc& srcEnd, BinaryOperation op, T neutral, cudaStream_t pStream = NULL)
    {
        nutty::base::Reduce(dst, srcBegin, Distance(srcBegin, srcEnd), op, neutral, pStream);
    }

    template <
        typename IteratorDst,
        typename IteratorSrc,
        typename BinaryOperation,
        typename T
    >
    __host__ void Reduce(IteratorDst& start, IteratorSrc& end, BinaryOperation op, T neutral, cudaStream_t pStream = NULL)
    {
        nutty::base::Reduce(start, start, Distance(start, end), op, neutral, pStream);
    }

    template <
        typename IteratorDst,
        typename IteratorSrc,
        typename BinaryOperation,
        typename T
    >
    __device__ void ReduceDP(IteratorDst& start, IteratorSrc& end, BinaryOperation op, T neutral, cudaStream_t pStream = NULL)
    {
        nutty::base::ReduceDP(start, start, Distance(start, end), op, neutral, pStream);
    }
}