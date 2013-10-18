#pragma once

namespace nutty
{
    namespace cuda
    {
        template <
            typename T
        >
        void Copy(T* dst, const T* src, size_t d, cudaMemcpyKind f)
        {
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaMemcpy(dst, src, d * sizeof(T), f));
        }
    }
}