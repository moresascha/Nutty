#pragma once
#include "Nutty.h"
#include <d3d11.h>
#include <cudaD3D11.h>
#include <cuda_gl_interop.h>
#include "cuda/Stream.h"

//remove this?
namespace nutty
{
    CUcontext g_CudaContext = NULL;
    CUdevice g_device = NULL;
    cudaStream_t g_currentStream = NULL;
    bool g_initialized = false;

    void SetStream(const nutty::cuStream& stream)
    {
        g_currentStream = (cudaStream_t)stream();
    }

    void SetDefaultStream(void)
    {
        g_currentStream = NULL;
    }

    bool Init(ID3D11Device* device /* = NULL */)
    {
        if(g_initialized)
        {
            return true;
        }

        g_initialized = true;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuInit(0));

        int devices = 0;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuDeviceGetCount(&devices));

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuDeviceGet(&g_device, 0));

        if(device)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuD3D11CtxCreate(&g_CudaContext, &g_device, CU_CTX_SCHED_BLOCKING_SYNC, device));
        }
        else
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuCtxCreate(&g_CudaContext, CU_CTX_SCHED_BLOCKING_SYNC, g_device));
        }

        return true;
    }
}