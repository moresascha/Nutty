#pragma once

namespace nutty
{
    namespace unary
    {
        template <
            typename T
        >
        struct Sequence
        {
            T i;
            Sequence(void) : i(0)
            {

            }

            T operator()(void)
            {
                return i++;
            }
        };

        template <
            typename T
        >
        struct Rand
        {
            __host__ T operator()(void)
            {
                return (T)rand();
            }
        };

        template <
            typename T
        >
        struct RandMax
        {
            T m_mod;
            RandMax(T max) : m_mod(max) {}
            __host__ T operator()(void)
            {
                return (T)rand() % m_mod;
            }
        };

        struct RandNorm
        {
            double operator()(void)
            {
                return (double)(rand() / (double)RAND_MAX);
            }
        };
    }

    namespace binary
    {
        template <
            typename T
        >
        struct Plus
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 + t1;
            }
        };

        template <
            typename T
        >
        struct Minus
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 - t1;
            }
        };

        template <
            typename T
        >
        struct Max
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 < t1 ? t1 : t0;
            }
        };

        template <
            typename T
        >
        struct Min
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 < t1 ? t0 : t1;
            }
        };

        template <
            typename T
        >
        struct Mul
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 * t1;
            }
        };

        template <
            typename T
        >
        struct Div
        {
            __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 / t1;
            }
        };
    }
}