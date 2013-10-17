#pragma once
#include "Inc.h"
#include "Stream.h"

namespace nutty
{
    class _cuda_record 
    {
        friend class nutty;
    private:
        int m_start, m_stop;
        cuStream* m_stream;
        float m_millis;
        _cuda_record(cuStream* stream) : m_millis(-1.0f), m_start(0), m_stop(0), m_stream(stream)
        {
            Create();
        }

        _cuda_record(void) : m_millis(-1.0f), m_start(0), m_stop(0), m_stream(0)
        {
            Create();
        }

        void Create(void)
        {
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventCreate(&m_start));
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventCreate(&m_stop)); 
        }

        void StartRecording(void)
        {
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventRecord(m_start, m_stream->m_stream));
        }

        void EndRecording(void)
        {
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventRecord(m_stop, m_stream->m_stream));
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventSynchronize(m_stop));
            //CUDA_RUNTIME_SAFE_CALLING(cudaEventElapsedTime(&m_millis, m_start, m_stop));
        }

        float GetElapsedTimeMS(void)
        {
            return m_millis;
        }

        void Destroy(void)
        {
            if(m_start)
            {
               // CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cudaEventDestroy(m_start));
            }

            if(m_stop)
            {
               // CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cudaEventDestroy(m_stop));
            } 

            m_start = 0;
            m_stop = 0;
        }
    };

    typedef _cuda_record* cuda_record;
}