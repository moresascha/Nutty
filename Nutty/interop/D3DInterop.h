#pragma once
#include "../Inc.h"
#include "DevicePtr.h"
#include "cuda_d3d11_interop.h"
#include "shared_resources.h"

struct ID3D11Buffer;

namespace nutty
{
    namespace interop
    {

        template <
            typename T
        >
        MappedBufferPtr<T> Wrap(ID3D11Buffer* pointer, unsigned int flags)
        {
            cudaGraphicsResource_t res;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsD3D11RegisterResource(&res, pointer, flags));
            return MappedBufferPtr<T>(res);
        }

        template <
            typename T
        >
        MappedTexturePtr<T> Wrap(ID3D11Texture2D* pointer, unsigned int flags)
        {
            cudaGraphicsResource_t res;
            CUDA_RT_SAFE_CALLING_NO_SYNC(cudaGraphicsD3D11RegisterResource(&res, pointer, flags));
            return MappedTexturePtr<T>(res);
        }
    }
}