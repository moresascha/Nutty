#include <ctime>
#include <profileapi.h>

namespace nutty
{
    class cuTimer
    {
    private:
        LARGE_INTEGER m_start;
        LARGE_INTEGER m_end;
        double m_freq;

        double m_timeMillis;
    public:
        cuTimer(void) : m_timeMillis(0)
        {
            VReset();
        }

        void Start(void)
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&m_start);
        }

        void Stop(void)
        {
            cudaDeviceSynchronize();
            QueryPerformanceCounter(&m_end);
        }

        double GetTime(double multiplier) const
        {
            return (double)((m_end.QuadPart - m_start.QuadPart) * multiplier) / m_freq;
        }

        double GetNanos(void) const
        {
            return GetTime(1e9);
        }

        double GetMicros(void) const
        {
            return GetTime(1e6);
        }

        double GetMillis(void) const
        {
            return GetTime(1e3);
        }

        double GetSeconds(void)
        {
            return GetTime(1);
        }

        void VTick(void)
        {
            Stop();
            m_timeMillis += GetMillis();
            Start();
        }

        void VReset(void)
        {
            m_timeMillis = 0;
            QueryPerformanceFrequency(&m_start);
            m_freq = (double)(m_start.QuadPart);
            Start();
        }

        float VGetFPS(void) const
        {
            return 0;
        }

        unsigned long VGetTime(void) const
        {
            return (unsigned long)(m_timeMillis + 0.5);
        }

        unsigned long VGetLastMillis(void) const
        {
            return (unsigned long)(GetMillis() + 0.5);
        }

        unsigned long VGetLastMicros(void) const
        {
            return (unsigned long)(GetMicros() + 0.5);
        }
    };
};