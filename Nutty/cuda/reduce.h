#pragma once

namespace nutty
{
    namespace cuda
    {
        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(T* dst, T* src, size_t d, BinaryOperation f)
        {
            
        }
    }
}