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
        typename Iterator_,
        typename Generator
    >
    void Fill(Iterator_& begin, Iterator_& end, Generator g)
    {
        nutty::base::Fill(begin, end, g);
    }
}