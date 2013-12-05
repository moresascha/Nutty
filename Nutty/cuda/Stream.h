#pragma once
#include "../Inc.h"
#include "Event.h"

namespace nutty
{
    class cuStream
    {
    private:
        CUstream m_stream;

        cuStream(cuStream&) {}

        std::vector<cuEvent> m_events;

    public:
        cuStream(void)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuStreamCreate(&m_stream, 0));
        }

        cuStream(cuStream&& s)
        {
            m_stream = s.m_stream;
            s.m_stream = NULL;
        }

        CUstream GetPointer(void) const
        {
            return m_stream;
        }

        CUstream operator()(void) const
        {
            return GetPointer();
        }

        void WaitEvent(cuEvent event)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuStreamWaitEvent(m_stream, event(), 0));
            m_events.push_back(std::move(event));
        }

        void ClearEvents(void)
        {
            m_events.clear();
        }

        cuEvent RecordEvent(void) const
        {
            cuEvent e;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuEventRecord(e(), m_stream));
            return e;
        }

        ~cuStream(void)
        {
            if(m_stream)
            {
                ClearEvents();
                CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuStreamDestroy(m_stream));
                m_stream = NULL;
            }
        }
    };

    class cuStreamPool
    {
    private:
        const static byte MAX_LIMIT = 16;
        cuStream* m_pStreams[MAX_LIMIT]; //fermi
        byte m_index;
        byte m_limit;

    public:
        cuStreamPool(byte limit = MAX_LIMIT) : m_index(0), m_limit(min(limit, MAX_LIMIT))
        {
            for(byte i = 0; i < m_limit; ++i)
            {
                m_pStreams[i] = new cuStream();
            }
        }

        cuStream& PeekNextStream(void)
        {
            return *m_pStreams[(m_index++) % m_limit];
        }

        ~cuStreamPool(void)
        {
            for(byte i = 0; i < m_limit; ++i)
            {
                SAFE_DELETE(m_pStreams[i]);
                m_pStreams[i] = NULL;
            }
        }
    };
}