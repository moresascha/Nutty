#pragma once
#include "base/reduce.h"

namespace nutty
{
    template <
        typename T
    >
    struct Max
    {
        T& operator()(T& t0, T& t1)
        {
            return t0 < t1 ? t1 : t0;
        }
    };

    template <
        typename T
    >
    struct Min
    {
        T& operator()(T& t0, T& t1)
        {
            return t0 > t1 ? t1 : t0;
        }
    };

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
        typename C,
        typename A,
        typename BinaryOperation
    >
    void Reduce(base::Base_Buffer<T, C, A>& data, BinaryOperation op)
    {
        Reduce(data.Begin(), data.End(), op);
    }

    template <
        typename T,
        typename C,
        typename A
    >
    void ReduceMax(base::Base_Buffer<T, C, A>& data)
    {
        Max<T> m;
        Reduce(data.Begin(), data.End(), m);
    }

    template <
        typename T,
        typename C,
        typename A
    >
    void ReduceMin(base::Base_Buffer<T, C, A>& data)
    {
        Min<T> m;
        Reduce(data.Begin(), data.End(), op, m);
    }
}