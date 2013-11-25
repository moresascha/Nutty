#pragma once
#include "../Inc.h"
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
        MappedPtr<T> Wrap(ID3D11Buffer* pointer, unsigned int flags)
        {
            cudaGraphicsResource_t res;
            cudaGraphicsD3D11RegisterResource(&res, pointer, flags);
            return MappedPtr<T>(res);
        }

        template <
            typename T
        >
        DevicePtr<T> Bind(MappedPtr<T>& mapped)
        {
            return mapped.Bind();
        }

        template <
            typename T
        >
        void Unbind(MappedPtr<T>& mapped)
        {
            mapped.Unbind();
        }
    }
}