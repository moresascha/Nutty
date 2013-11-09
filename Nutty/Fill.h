#pragma once
#include "base/Buffer.h"
#include "base/fill.h"
#include "Copy.h"

namespace nutty
{
    template <
        typename T
    >
    struct DefaultGenerator
    {
        T v;
        DefaultGenerator(T vv) : v(vv) {}

        const T& operator()() { return v; }
    };

    template <
        typename T,
        typename C,
        typename Generator
    >
    void Fill(Iterator<T, C>& begin, Iterator<T, C>& end, Generator g)
    {
        nutty::base::Fill(begin, end, g);
    }

    template <
        typename T,
        typename C
    >
    void Fill(Iterator<T, C>& begin, Iterator<T, C>& end, const T& v)
    {
        nutty::base::Fill(begin, end, v);
    }
}