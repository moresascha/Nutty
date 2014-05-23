#include "test_help.cuh"
#include "../Sort.h"
#include "../Inc.h"
#include "../ForEach.h"
#include "../Functions.h"

int main(void)
{
    #define _CRTDBG_MAP_ALLOC

#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    nutty::Init();

    nutty::DeviceBuffer<uint> a(32);

    RNG rng;
    nutty::Fill(a.Begin(), a.End(), rng);

    printBuffer(a);

    nutty::Sort(a.Begin(), a.End(), nutty::BinaryDescending<int>());

    printBuffer(a);

    nutty::Release();

    return 0;
}