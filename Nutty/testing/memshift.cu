#include "test_help.cuh"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Scan.h"
#include "../Functions.h"

int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    //create nutty
    nutty::Init(); 

    nutty::DeviceBuffer<int> data(20);

    nutty::Fill(data.Begin(), data.End(), nutty::unary::Sequence<int>());

    for(int i = 0; i < data.Size(); ++i)
    {
        print(data[i]);
    }

    OutputDebugStringA("\n");

    //nutty::cuda::shiftMemory<int, 1><<<2, 10, 10 * sizeof(int)>>>(data.Begin()(), data.Size());

    for(int i = 0; i < data.Size(); ++i)
    {
        print(data[i]);
    }

    //release nutty
    nutty::Release();

    return 0;
}