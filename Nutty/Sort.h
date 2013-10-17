#pragma once
#include "base/sort.h"

namespace nutty
{
    template <
        typename T,
        typename C,
        typename BinaryOperation
    >
    void Sort(Iterator<T, C>& begin, Iterator<T,C>& end, BinaryOperation op)
    {
        nutty::base::Sort(begin, end, op);
    }

    template <
        typename T,
        typename C,
        typename A,
        typename BinaryOperation
    >
    void Sort(base::Base_Buffer<T, C, A>& data, BinaryOperation op)
    {
       nutty::Sort(data.Begin(), data.End(), op);
    }

    template <
        typename T,
        typename C,
        typename A
    >
    void SortDescending(base::Base_Buffer<T, C, A>& data)
    {
        BinaryDescending<T> op;
        Sort(data, op);
    }

    template <
        typename T,
        typename C,
        typename A
    >
    void SortAscending(base::Base_Buffer<T, C, A>& data)
    {
        BinaryAscending<T> op;
        Sort(data, op);
    }
}