#include "test_help.cuh"

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

    nutty::DeviceBuffer<int> data(2033);
    nutty::DeviceBuffer<int> prefixSum(data.Size());
    nutty::DeviceBuffer<int> sums(4);
    nutty::ZeroMem(sums);
    nutty::ZeroMem(prefixSum);

    nutty::Fill(data.Begin(), data.End(), BinaryRNG());

    data.Insert(data.Size()-1, 1);

    printBuffer(data);

    nutty::InclusiveScan(data.Begin(), data.End(), prefixSum.Begin(), sums.Begin(), nutty::PrefixSumOp<int>());

    printBuffer(prefixSum);
    printBuffer(sums);

    nutty::ZeroMem(prefixSum);
    nutty::ZeroMem(sums);

    nutty::ExclusiveScan(data.Begin(), data.End(), prefixSum.Begin(), sums.Begin(), nutty::PrefixSumOp<int>());

    printBuffer(prefixSum);
    printBuffer(sums);

    //release nutty
    nutty::Release();

    return 0;
}