#pragma once
#include "Inc.h"

#ifdef _DEBUG
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart.lib")
#else
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart.lib")
#endif

struct ID3D11Device;

namespace nutty
{
    class cuModule;
    class cuKernel;
    struct map_info 
    {
        size_t size;
        CUdeviceptr ptr;
    };

    enum ArrayFormat
    {
        eR,
        eRG,
        eRGB,
        eRGBA
    };

    struct TextureBindung
    {
        std::string m_textureName;
        //cuArray m_cudaArray;
        CUfilter_mode m_filterMode;
        CUaddress_mode m_addressMode;
        uint m_flags;
    };

    CUcontext g_CudaContext;
    CUdevice g_device = NULL;
    bool initialized = false;

    std::map<std::string, cuModule*> g_modules;
    std::map<std::string, cuKernel*> g_kernel;

    bool Init(ID3D11Device* device = NULL)
    {
        if(initialized)
        {
            return true;
        }

        initialized = true;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuInit(0));

        int devices = 0;

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuDeviceGetCount(&devices));

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuDeviceGet(&g_device, 0));

        if(device)
        {
//            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuD3D11CtxCreate(&g_CudaContext, &g_device, CU_CTX_SCHED_BLOCKING_SYNC, device));
        }
        else
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuCtxCreate(&g_CudaContext, CU_CTX_SCHED_BLOCKING_SYNC, g_device));
        }

        return true;
    }

    cuKernel* GetKernel(const char* name);

    cuModule* GetModule(void);

    void Release(void)
    {
        if(g_CudaContext)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuCtxDestroy(g_CudaContext));
            g_CudaContext = NULL;
        }
    }

}

