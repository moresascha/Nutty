#pragma once
#include "interop/D3DInterop.h"
#include "DeviceBuffer.h"

namespace nutty
{
    template <
        typename T,
        typename GFXBuffer
    >
    MappedBufferPtr<T> WrapBuffer(GFXBuffer* pointer, uint flags = 0)
    {
        return interop::Wrap<T>(pointer, flags);
    }

    template <
        typename T,
        typename GFXTexture
    >
    MappedTexturePtr<T> WrapTexture(GFXTexture* pointer, uint flags = 0)
    {
        return interop::Wrap<T>(pointer, flags);
    }
}