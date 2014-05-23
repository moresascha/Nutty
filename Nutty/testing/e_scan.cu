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
    nutty::DeviceBuffer<int> prefixSum(data.Size());
    nutty::DeviceBuffer<int> sums(data.Size());

    nutty::Fill(data.Begin(), data.End(), BinaryRNG());

    for(int i = 0; i < data.Size(); ++i)
    {
        print(data[i]);
    }

    OutputDebugStringA("\n");

    nutty::ExclusiveScan(data.Begin(), data.End(), prefixSum.Begin(), sums.Begin(), nutty::PrefixSumOp<int>());

    for(int i = 0; i < data.Size(); ++i)
    {
        print(prefixSum[i]);
    }

    //release nutty
    nutty::Release();

    return 0;
}