#pragma once
#include "../Inc.h"

namespace nutty
{
    class cuEvent
    {
    private:
        CUevent m_event;

        cuEvent(const cuEvent& ) {}

    public:
        cuEvent(void)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuEventCreate(&m_event, 0));
        }

        cuEvent(cuEvent&& event)
        {
            m_event = event.m_event;
            event.m_event = NULL;
        }

        CUevent operator()(void) const
        {
            return GetPointer();
        }

        CUevent GetPointer(void) const
        {
            return m_event;
        }

        CUevent Free(void)
        {
            CUevent cpy = m_event;
            m_event = NULL;
            return cpy;
        }

        cuEvent& operator=(cuEvent&& other)
        {
            m_event = other.m_event;
            other.m_event = NULL;
            return *this;
        }

        ~cuEvent(void)
        {
            if(m_event)
            {
                CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuEventDestroy(m_event));
                m_event = NULL;
            }
        }
    };
}