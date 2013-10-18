#pragma once
#include "../Inc.h"

namespace nutty
{
    class cuStream
    {
    private:
        CUstream m_stream;

    public:
        cuStream(uint flags = 0)
        {
            CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamCreate(&m_stream, flags));
        }

        const CUstream& GetPointer(void) const
        {
            return m_stream;
        }

        ~cuStream(void)
        {
            CUDA_DRIVER_SAFE_CALLING_SYNC(cuStreamDestroy(m_stream));
        }
    };
}