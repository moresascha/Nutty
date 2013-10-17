#pragma once
#include "Inc.h"
#include "Resource.h"

namespace nutty
{
    class _cuda_array : public cuResource
    {
        friend class nutty;
    protected:
        _cuda_array(CUarray array) : m_array(array)
        {

        }

        ResType VGetType(VOID) CONST
        {
            return eARRAY;
        }

        VOID VDestroy(VOID);
    public:
        UINT VGetByteCount(VOID) CONST { return byteSize; }
        UINT VGetElementCount(VOID) CONST { return elements; }
        CUarray m_array;
    };

    typedef _cuda_array* cuda_array;
}