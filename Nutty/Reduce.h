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
        typename T,
        typename C,
        typename BinaryOperation
    >
    void Reduce(Iterator<T, C>& begin, Iterator<T, C>& end, BinaryOperation op)
    {
        nutty::base::Reduce(begin, end, op);
    }

    template <
        typename T,
        typename C0,
        typename C1,
        typename A0,
        typename A1,
        typename BinaryOperation
    >
    void Reduce(base::Base_Buffer<T, C0, A0>& dst, base::Base_Buffer<T, C1, A1>& src, BinaryOperation op)
    {
        nutty::base::Reduce(dst.Begin(), dst.End(), src.Begin(), src.End(), op);
    }

    template <
        typename T,
        typename C,
        typename A,
        typename BinaryOperation
    >
    void Reduce(base::Base_Buffer<T, C, A>& data, BinaryOperation op)
    {
        nutty::base::Reduce(data.Begin(), data.End(), op);
    }

    template <
        typename T,
        typename C,
        typename A,
        typename BinaryOperation
    >
    T Reduce(BinaryOperation op, base::Base_Buffer<T, C, A>& data)
    {
        return nutty::base::Reduce(op, data.Begin(), data.End());
    }
}