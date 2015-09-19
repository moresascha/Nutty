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

        template <
            typename T
        >
        struct RandNorm
        {
            T m_scale;
            RandNorm(T scale)
            {
                m_scale = scale;
            }
            RandNorm(void)
            {
                m_scale = 1;
            }
            T operator()(void)
            {
                return m_scale * (T)(rand() / (T)RAND_MAX);
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
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 + t1;
            }
        };

        template <
            typename T
        >
        struct Minus
        {
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 - t1;
            }
        };

        template <
            typename T
        >
        struct Max
        {
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 < t1 ? t1 : t0;
            }
        };

        template <
            typename T
        >
        struct Min
        {
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 < t1 ? t0 : t1;
            }
        };

        template <
            typename T
        >
        struct Mul
        {
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 * t1;
            }
        };

        template <
            typename T
        >
        struct Div
        {
            __forceinline__ __device__ __host__ T operator()(T t0, T t1)
            {
                return t0 / t1;
            }
        };
    }
}