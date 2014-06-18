#pragma once
#include "Inc.h"

#ifdef _DEBUG
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart_static.lib")
    //#pragma comment(lib, "cudart.lib")
    #pragma comment(lib, "cudadevrt.lib")
    //#pragma comment(lib, "Nuttyx64Debug.lib")
    //#pragma comment(lib, "cudart.lib")
#else
    #pragma comment(lib, "cuda.lib")
    #pragma comment(lib, "cudart_static.lib")
    #pragma comment(lib, "cudart.lib")
    #pragma comment(lib, "cudadevrt.lib")
    //#pragma comment(lib, "Nuttyx64Release.lib")
    //#pragma comment(lib, "cudart.lib")
#endif

struct ID3D11Device;

namespace nutty
{
    class cuModule;
    class cuKernel;
    class cuStream;
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

    extern CUcontext g_CudaContext;
    extern CUdevice g_device;
    extern bool g_initialized;
    extern cudaStream_t g_currentStream;

    bool Init(ID3D11Device* device = NULL);

    __inline void Release(void)
    {
        if(g_CudaContext)
        {
            cudaDeviceReset();
            //CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuCtxDestroy(g_CudaContext));
            g_CudaContext = NULL;
        }
    }

    void SetStream(const cuStream& stream);

    void SetDefaultStream(void);
}

