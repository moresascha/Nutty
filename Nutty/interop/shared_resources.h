#pragma once
#include "../Inc.h"

namespace nutty
{
    template <
        typename T
    >
    class MappedPtr
    {
    private:
        cudaGraphicsResource_t m_res;
        size_t m_size;

        MappedPtr(MappedPtr&);

    public:
        MappedPtr(cudaGraphicsResource_t res) : m_res(res), m_size(0)
        {

        }

        MappedPtr(MappedPtr&& res)
        {
            m_res = res.m_res;
            m_size = 0;
            res.m_res = NULL;
        }

        cudaGraphicsResource_t Get(void)
        {
            return m_res;
        }

        DevicePtr<T> Bind(void)
        {
            cudaGraphicsResource_t res = Get();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &res));
            T* devptr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&devptr, &m_size, res));
            return DevicePtr<T>(devptr);
        }

        void Unbind(void)
        {
            cudaGraphicsResource_t res = Get();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &res));
        }

        size_t Size(void)
        {
            return m_size;
        }

        ~MappedPtr(void)
        {
            CHECK_CNTX();
            if(m_res)
            {
                CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnregisterResource(m_res));
            }
        }
    };

    class ScopedBind
    {

    };
}