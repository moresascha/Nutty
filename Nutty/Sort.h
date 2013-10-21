#pragma once
#include "base/sort.h"

namespace nutty
{
    template <
        typename K,
        typename T,
        typename C,
        typename C1,
        typename BinaryOperation
    >
    void Sort(Iterator<K, C>& keyBegin, Iterator<K, C>& keyEnd, Iterator<T, C1>& values, BinaryOperation op)
    {
        nutty::base::Sort(keyBegin, keyEnd, values, op);
    }

    template <
        typename T,
        typename C,
        typename BinaryOperation
    >
    void Sort(Iterator<T, C>& begin, Iterator<T, C>& end, BinaryOperation op)
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