#pragma once
#include "interop/D3DInterop.h"
#include "DeviceBuffer.h"

namespace nutty
{
    template <
        typename T,
        typename GFXBuffer
    >
    MappedPtr<T> Wrap(GFXBuffer* pointer)
    {
        return interop::Wrap<T>(pointer, cudaGraphicsMapFlagsNone);
    }
}