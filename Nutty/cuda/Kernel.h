#pragma once
#include "../Inc.h"
#include "Stream.h"
#include "../DeviceBuffer.h"
#include "../DevicePtr.h"

namespace nutty
{
    class cuKernel
    {
    private:
        CUfunction m_fpCuda;
        std::string m_func_name;
        uint m_argCount;
        dim3 m_gridDim;
        dim3 m_blockDim;
        uint m_shrdMemBytes;
        void* m_ppExtras[32]; //max 32 arguments
        void* m_ppArgs[32];

    public:
        cuKernel(CUfunction function) : m_shrdMemBytes(0), m_argCount(0)
        {
            memset(m_ppArgs, 0, 32 * sizeof(void*));
            memset(m_ppExtras, 0, 32 * sizeof(void*));
            Create(function);
        }

        cuKernel(void) : m_fpCuda(NULL), m_shrdMemBytes(0), m_argCount(0)
        {
            memset(m_ppArgs, 0, 32 * sizeof(void*));
            memset(m_ppExtras, 0, 32 * sizeof(void*));
        }

        void Create(CUfunction function)
        {
            m_fpCuda = function;
        }

        void SetSharedMemory(uint bytes)
        {
            m_shrdMemBytes = bytes;
        }

        void SetDimension(const dim3& grid, const dim3& block)
        {
            m_gridDim = grid;
            m_blockDim = block;
        }

        template<
            typename T
        >
        void SetKernelArg(uint index, T& ptr)
        {
            m_ppArgs[index] = (void**)&ptr;
                /*
#ifdef _DEBUG
            assert(index <= m_argCount);
#endif
            if(index < m_argCount)
            {
                
                return;
            }

            m_argCount++;
        
            void** tmp = new void*[m_argCount];

            for(uint i = 0; i < m_argCount-1; ++i)
            {
                tmp[i] = m_ppArgs[i];
            }

            tmp[m_argCount-1] = ptr;

            if(m_ppArgs)
            {
                delete[] m_ppArgs;
            }

            m_ppArgs = tmp; */
        }

        void SetRawKernelArg(uint index, void** ptr)
        {
            m_ppArgs[index] = ptr;
        }

        template< 
            typename T
        >
        void SetKernelArg(uint index, nutty::DevicePtr<T>& arg)
        {
            CUdeviceptr* ptr = (CUdeviceptr*)arg.GetRawPointerPtr();
            m_ppArgs[index] = ptr;
        }

        template< 
            typename T
        >
        void SetKernelArg(uint index, nutty::DeviceBuffer<T>& arg)
        {
            CUdeviceptr* ptr = (CUdeviceptr*)arg.GetRawPointer();
            m_ppArgs[index] = ptr;
        }

        void Call(const cuStream& stream)
        {
            Call(stream.GetPointer());
        }

        void Call(const CUstream stream = NULL)
        {
#ifdef _DEBUG
            for(byte i = 0; i < 16; ++i)
            {
                if(m_ppArgs[max(2 * i,0)] == NULL && m_ppArgs[max(2 * i + 1,0)] != NULL)
                {
                    assert(0 && L"Invalid Kernel Arguments!");
                }
            }
#endif
            CUDA_DRIVER_SAFE_CALLING_SYNC(
                cuLaunchKernel(
                m_fpCuda, 
                m_gridDim.x, m_gridDim.y, m_gridDim.z, 
                m_blockDim.x, m_blockDim.y, m_blockDim.z,
                m_shrdMemBytes, stream, m_ppArgs, m_ppExtras)
                );
        }

        ~cuKernel(void)
        {
        }
    };
}