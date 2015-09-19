#pragma once
#define NUTTY_DEBUG
#include "Nutty.h"
#include <d3d11.h>
#include "inc/helper_cuda.h"
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

        CUDA_RT_SAFE_CALLING_NO_SYNC(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

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

    void __printf(char* format, ...)
    {
        int len = (int)strlen(format);
        char buffer[2048];

        va_list ap;
        va_start(ap, format);

        vsprintf_s(buffer, format, ap);

        va_end(ap);
        OutputDebugStringA(buffer);
    }

    void PrintDeviceInfos(void)
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0)
        {
            __printf("There are no available device(s) that support CUDA\n");
        }
        else
        {
            __printf("Detected %d CUDA Capable device(s)\n", deviceCount);
        }

        int dev, driverVersion = 0, runtimeVersion = 0;
        for (dev = 0; dev < deviceCount; ++dev)
        {
            cudaSetDevice(dev);
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, dev);

            __printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

            // Console log
            cudaDriverGetVersion(&driverVersion);
            cudaRuntimeGetVersion(&runtimeVersion);
            __printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
            __printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

            char msg[256];
            sprintf_s(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
            __printf("%s", msg);

            __printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
                deviceProp.multiProcessorCount,
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
                _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
            __printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
            // This is supported in CUDA 5.0 (runtime API device properties)
            __printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
            __printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

            if (deviceProp.l2CacheSize)
            {
                __printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
            }

#else
            // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
            int memoryClock;
            getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
            __printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
            int memBusWidth;
            getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
            __printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
            int L2CacheSize;
            getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

            if (L2CacheSize)
            {
                __printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
            }

#endif

            __printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
                deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
                deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
            __printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
                deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
            __printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
                deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


            __printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
            __printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
            __printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
            __printf("  Warp size:                                     %d\n", deviceProp.warpSize);
            __printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
            __printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
            __printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxThreadsDim[0],
                deviceProp.maxThreadsDim[1],
                deviceProp.maxThreadsDim[2]);
            __printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
                deviceProp.maxGridSize[0],
                deviceProp.maxGridSize[1],
                deviceProp.maxGridSize[2]);
            __printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
            __printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
            __printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
            __printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
            __printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
            __printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
            __printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
            __printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
            __printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
            __printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
            __printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

            const char *sComputeMode[] =
            {
                "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
                "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
                "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
                "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
                "Unknown",
                NULL
            };
            __printf("  Compute Mode:\n");
            __printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
        }
    }
}