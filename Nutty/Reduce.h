#pragma once
#include "base/reduce.h"

namespace nutty
{
    template <
        typename T,
        typename Operation
    >
    struct BinaryOP
    {
        Operation m_op;
        BinaryOP(Operation op) : m_op(op)
        {

        }
        __device__ __host__ T& operator()(T& t0, T& t1)
        {
            return m_op(t0, t1);
        }
    };

    template <
        typename T,
        typename C,
        typename BinaryOperation
    >
    void Reduce(Iterator<T, C>& begin, Iterator<T, C>& end, BinaryOperation op)
    {
        BinaryOP<T, BinaryOperation> _op(op);
        nutty::base::Reduce(begin, end, _op);
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
        BinaryOP<T, BinaryOperation> _op(op);
        nutty::base::Reduce(dst.Begin(), dst.End(), src.Begin(), src.End(), _op);
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
}