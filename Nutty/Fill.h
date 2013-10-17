#pragma once
#include "base/Buffer.h"
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
    void Generate(Iterator<T, C>& begin, Iterator<T, C>& end, Generator g)
    {
        size_t d = Distance(begin, end);

        nutty::HostBuffer<T> b(d);

        for(auto s = b.Begin(); s != b.End(); ++s)
        {
            s = g();
        }

        Copy(begin, b.Begin(), d);
    }

    template <
        typename T,
        typename C
    >
    void Fill(Iterator<T, C>& begin, Iterator<T, C>& end, T v)
    {
        DefaultGenerator<T> g(v);
        Generate(begin, end, g);
    }
}