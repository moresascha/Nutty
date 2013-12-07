#pragma once
#include "../Inc.h"

namespace nutty
{
    template <
        typename T
    >
    class MappedPtr
    {
    protected:
        cudaGraphicsResource_t m_res;
        size_t m_size;

        MappedPtr(MappedPtr&);

        MappedPtr& operator=(MappedPtr&);

        void _delete(void)
        {
            CHECK_CNTX();
            if(m_res)
            {
                CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnregisterResource(m_res));
                m_res = NULL;
            }
        }

    public:
        MappedPtr(cudaGraphicsResource_t res) : m_res(res), m_size(0)
        {

        }
// 
//         MappedPtr& operator=(MappedPtr&& res)
//         {
//             if(&res != this)
//             {
//                 _delete();
//                 m_res = res.m_res;
//                 m_size = res.m_size;
//                 res.m_res = NULL;
//             }
//             return *this;
//         }

        MappedPtr(MappedPtr&& res)
        {
            m_res = res.m_res;
            m_size = res.m_size;
            res.m_res = NULL;
        }

        MappedPtr(void)
        {
            m_res = NULL;
            m_size = 0;
        }

        cudaGraphicsResource_t Get(void)
        {
            return m_res;
        }

        void Unbind(cudaStream_t stream = NULL)
        {
            cudaGraphicsResource_t res = Get();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsUnmapResources(1, &res, stream));
        }

        size_t Size(void)
        {
            return m_size;
        }

        virtual ~MappedPtr(void)
        {
            _delete();
        }
    };

    template <
        typename T
    >
    class MappedBufferPtr : public MappedPtr<T>
    {
        typedef MappedPtr<T> base_class;

    public:
        MappedBufferPtr(cudaGraphicsResource_t res) : base_class(res)
        {

        }

        MappedBufferPtr& operator=(MappedBufferPtr&& res)
        {
            if(&res != this)
            {
                _delete();
                m_res = res.m_res;
                m_size = res.m_size;
                res.m_res = NULL;
            }
            return *this;
        }

        MappedBufferPtr(MappedBufferPtr&& res) : base_class(std::move(res))
        {
        }

        MappedBufferPtr(void) : base_class()
        {
        }

        DevicePtr<T> Bind(cudaStream_t stream = NULL)
        {
            cudaGraphicsResource_t res = Get();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &res, stream));
            T* devptr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsResourceGetMappedPointer((void**)&devptr, &m_size, res));
            return DevicePtr<T>(devptr);
        }
    };

    template <
        typename T
    >
    class MappedTexturePtr : public MappedPtr<T>
    {
        typedef MappedPtr<T> base_class;

    public:
        MappedTexturePtr(cudaGraphicsResource_t res) : base_class(res)
        {

        }

        MappedTexturePtr& operator=(MappedTexturePtr&& res)
        {
            if(&res != this)
            {
                _delete();
                m_res = res.m_res;
                m_size = res.m_size;
                res.m_res = NULL;
            }
            return *this;
        }

        MappedTexturePtr(MappedTexturePtr&& res) : base_class(std::move(res))
        {
        }

        MappedTexturePtr(void) : base_class()
        {
        }

        DevicePtr<T> Bind(cudaStream_t stream = NULL)
        {
            cudaGraphicsResource_t res = Get();
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsMapResources(1, &res, stream));
            T* devptr;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsSubResourceGetMappedArray((cudaArray_t*)&devptr, res, 0, 0));
            return DevicePtr<T>(devptr);
        }
    };
}