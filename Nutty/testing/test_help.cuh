#pragma once
#include <windows.h>
#ifdef _DEBUG
#define NUTTY_DEBUG
#endif
#include "../Nutty.h"
#include "../Fill.h"
#include <sstream>

template <
    typename Buffer
>
void printBuffer(const Buffer& buffer, size_t max = -1, const char* trim = " ")
{
    std::stringstream ss;
    for(uint i = 0; i < min(max, buffer.Size()); ++i)
    {
        ss << buffer[i] << trim;
    }
    ss << "\n";
    OutputDebugStringA(ss.str().c_str());
}

template <
    typename T
>
void print(const T& t)
{
    std::stringstream ss;
    ss << t;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
}

struct RNG
{
    int _modulo;
    RNG(int seed = 10, int modulo = 100) : _modulo(modulo)
    {
        srand(seed);
    }

    int operator()()
    {
        return rand() % _modulo;
    }
};

struct BinaryRNG : public RNG
{
    BinaryRNG(int seed = 10) : RNG(10, 2)
    {
        srand(seed);
    }
};