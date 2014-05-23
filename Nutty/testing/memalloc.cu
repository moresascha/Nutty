#include "test_help.cuh"
#include "../DeviceBuffer.h"
#include <cuda_runtime.h>

//__global__ 

int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    //create nutty
    nutty::Init();

    size_t freeMemory, totalMemory;
    cudaMemGetInfo(&freeMemory, &totalMemory);

    printf("%u %u\n", totalMemory, freeMemory);

    nutty::DeviceBuffer<char> memory0(1024 * 1024 * 2);
    nutty::DeviceBuffer<char> memory1(1024 * 1024 * 1024);
    nutty::DeviceBuffer<char> memory2(1024 * 1024 * 1024);

    nutty::Fill(memory0.Begin(), memory0.End(), (char)0);
    nutty::Fill(memory1.Begin(), memory1.End(), (char)0);

    //release nutty
    nutty::Release();

    return 0;
}