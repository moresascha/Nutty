#include <cutil_math.h>
#include "test_help.cuh"
#include "../cuda/Globals.cuh"

template<>
struct ShrdMemory<uint4>
{
    __device__ uint4* Ptr(void) 
    { 
        extern __device__ __shared__ uint4 s_int4[];
        return s_int4;
    }
};

#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Scan.h"
#include "../Functions.h"

struct PrefixSum4Op
{
    __device__ uint4 operator()(uint4 elem)
    {
        return elem;
    }

    __device__ __host__ uint4 GetNeutral(void)
    {
        uint4 v = {0};
        return v;
    }
};

struct BinaryRNG4
{
    BinaryRNG rr;

    uint4 operator()()
    {
        uint4 v;
        v.x = rr();
        v.y = rr();
        v.z = rr();
        v.w = rr();
        return v;
    }
};

bool operator==(const uint4& v0, const uint4& v1)
{
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
}

int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    //create nutty
    nutty::Init(); 
    uint elemsPerBlock = 512;
    uint elementCount = 1024 * 1024; 

    nutty::DeviceBuffer<uint4> data(elementCount);
    nutty::HostBuffer<uint4> h_data(elementCount);
    nutty::HostBuffer<uint4> h_prefixSum(elementCount);
    nutty::HostBuffer<uint4> h_devPrefixSum(elementCount);
    nutty::DeviceBuffer<uint4> prefixSum(elementCount);
    nutty::DeviceBuffer<uint4> sums(elementCount);
    nutty::DeviceBuffer<uint4> scannedSums(elementCount);

    nutty::Fill(h_data.Begin(), h_data.End(), BinaryRNG4());
    nutty::Copy(data.Begin(), h_data.Begin(), h_data.End());
    nutty::ZeroMem(prefixSum);

    uint4 sum = {0};
    for(int i = 0; i < data.Size(); ++i)
    {
        h_prefixSum.Insert(i, sum);
        sum.x += h_data[i].x;
        sum.y += h_data[i].y;
        sum.z += h_data[i].z;
        sum.w += h_data[i].w;
    }

    nutty::TScanner<uint4, PrefixSum4Op> test;
    test.Resize(elementCount);
    test.ExcScan(data.Begin(), data.End(), PrefixSum4Op());

    //nutty::PrintDeviceInfos();
    
    OutputDebugStringA("\n");

    size_t grid;
    //nutty::cuda::ScanPerBlock(data.Begin()(), prefixSum.Begin()(), sums.Begin()(), elementCount, PrefixSum4Op(), 0, &grid);
    nutty::PrefixSumOp<uint> _op;
//    m_pSumScanner->ExcScan(m_sums.Begin(), m_sums.End(), _op);
//    nutty::cuda::_spreadSums(m_scannedData.Begin()(), m_pSumScanner->GetPrefixSum().Begin()(), grid - 1, m_scannedData.Size());

    //nutty::ExclusiveScan(data.Begin(), data.End(), prefixSum.Begin(), sums.Begin(), scannedSums.Begin(), PrefixSum4Op());

    DEVICE_SYNC_CHECK();

    //nutty::Copy(h_devPrefixSum.Begin(), prefixSum.Begin(), prefixSum.End());
    nutty::Copy(h_devPrefixSum.Begin(), test.GetPrefixSum().Begin(), test.GetPrefixSum().End());

    DEVICE_SYNC_CHECK();

    for(int i = 0; i < data.Size(); ++i)
    {
        if(!(h_prefixSum[i] == h_devPrefixSum[i]))
        {
            std::stringstream ss;
            ss << "Index=" << i << "\n";
            ss << h_prefixSum[i].x << " " << h_prefixSum[i].y << " " << h_prefixSum[i].z << " " << h_prefixSum[i].w << "!=";
            ss << h_devPrefixSum[i].x << " " << h_devPrefixSum[i].y << " " << h_devPrefixSum[i].z << " " << h_devPrefixSum[i].w << "\n";
            //ss << i << " - Error\n";
            OutputDebugStringA(ss.str().c_str());
            exit(0);
        }
    }

    OutputDebugStringA("\n");

    //release nutty
    nutty::Release();

    return 0;
}