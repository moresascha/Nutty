#include "test_help.cuh"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Scan.h"
#include "../Functions.h"
#include "../cuTimer.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <map>

struct PrefixSumOp4
{
    __device__ int4 operator()(int4 elem)
    {
        return elem;
    }

    __device__ __host__ int4 GetNeutral(void)
    {
        int4 a = {0,0,0,0};
        return a;
    }
};

float g_percent;

byte rr(void)
{
    return (byte)(rand()/(float)RAND_MAX < g_percent ? 1 : 0);
}

std::map<uint, double> g_times;

template<bool mine, bool check>
void test(float percentOnes, uint elementCount, uint elemsPerBlock = 2048)
{
    g_percent = percentOnes;

    nutty::DeviceBuffer<byte> data(elementCount);
    nutty::DeviceBuffer<uint> scanned(elementCount);
    nutty::DeviceBuffer<uint> sums(elementCount);
    nutty::HostBuffer<byte> h_data(elementCount);
    nutty::HostBuffer<uint> h_prefixSum(elementCount);
    nutty::HostBuffer<uint> h_devPrefixSum(elementCount);

    nutty::Fill(h_data.Begin(), h_data.End(), rr);
    nutty::Copy(data.Begin(), h_data.Begin(), h_data.End());

    nutty::cuTimer timer;
    uint sum = 0;
    for(int i = 0; i < data.Size(); ++i)
    {
        h_prefixSum.Insert(i, sum);
        sum += h_data[i];
    }

//     nutty::Scanner test;
//     test.Resize(elementCount);

    srand(0);
    thrust::device_vector<int> dev_vec(elementCount);
    thrust::host_vector<int>   host_vec(elementCount);
   // thrust::generate(host_vec.begin(), host_vec.end(), rr);

    dev_vec = host_vec;
    const uint BLOCK_SIZE = 256;
    const uint blockCount = 1;
    const uint elemsPerThread = 4 * elemsPerBlock / BLOCK_SIZE;
    const uint scannedElemsPerBlock = elemsPerThread * BLOCK_SIZE;
    uint grid = nutty::cuda::GetCudaGrid(elementCount, scannedElemsPerBlock);
    timer.Start();
    int times = 32;

    for(int i = 0;  i < times; ++i)
    {
        if(!mine)
        {
            thrust::exclusive_scan(dev_vec.begin(), dev_vec.end(), dev_vec.begin());
        }
        else
        {
            //test.ExcBinaryScan(data.Begin(), data.End(), PrefixSumOp4());
            __binaryGroupScanN<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>((const uchar4*)data.GetConstPointer(), (uint4*)scanned.GetPointer(), sums.GetPointer(), elementCount, elemsPerBlock);
            //__binaryGroupScan4Test<BLOCK_SIZE, 2 * BLOCK_SIZE><<<grid, BLOCK_SIZE>>>((uchar4*)data.GetConstPointer(), (uint4*)scanned.GetPointer(), sums.GetPointer(), elementCount);
            //__binaryGroupScan4<uchar4, BLOCK_SIZE, 2 * BLOCK_SIZE><<<grid, BLOCK_SIZE>>>((uchar4*)data.GetConstPointer(), (uint4*)scanned.GetPointer(), sums.GetPointer(), elementCount);

           // __binaryGroupScan4B<BLOCK_SIZE, 2 * BLOCK_SIZE><<<grid, BLOCK_SIZE>>>((uint4*)data.GetConstPointer(), (uint16*)scanned.GetPointer(), sums.GetPointer(), elementCount);
// 
//             switch(grid)
//             {
//             case 1    :
//             case 2    :
//             case 4    :
//             case 8    :
//             case 16   : 
//             case 32   :
//             case 64   : 
//             case 128  : 
//             case 256  : 
//             case 512  : 
//             case 1024 : __binaryGroupScan4<uint4, 128, 256><<<1, 128>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), NULL, grid); break;
//             case 2048 : __binaryGroupScan4<uint4, 256, 512><<<1, 256>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), NULL, grid); break;
//             case 4096 : __binaryGroupScan4<uint4, 512, 1024><<<1, 512>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), NULL, grid); break;
//             case 8192 : __binaryGroupScan4<uint4, 1024, 2048><<<1, 1024>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), NULL, grid); break;
//             default : OutputDebugStringA("error in sumscount\n"); exit(0);
//             }

            if(grid > 1)
            {
                /*
                if(grid >= 8192)
                {
                    __groupScan4TestAll<1024, 2048><<<1, 1024>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), grid);
                }
                else if(grid >=4096)
                {
                    __groupScan4TestAll<512, 1024><<<1, 512>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), grid);
                }
                else if(grid >=2048)
                {
                    __groupScan4TestAll<256, 512><<<1, 256>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), grid);
                }
                else
                {
                    __groupScan4TestAll<128, 256><<<1, 256>>>((uint4*)sums.GetConstPointer(), (uint4*)sums.GetPointer(), grid);
                }*/
                nutty::PrefixSumOp<uint> _op;
                __completeScan<1024><<<1, 1024>>>(sums.GetConstPointer(), sums.GetPointer(), _op, grid);

                uint k = 2;
                uint elems = (elementCount - BLOCK_SIZE * elemsPerThread + (elementCount%2)) / k;
                //uint elems = (elementCount - 2*BLOCK_SIZE + (elementCount%2)) / k;
                //N /= 4;

                uint g = nutty::cuda::GetCudaGrid(elems, BLOCK_SIZE);
                //__spreadScannedSumsSingleT<<<g, BLOCK_SIZE>>>((uint*)scanned.GetPointer(), (uint*)sums.GetConstPointer(), elementCount);
                __spreadScannedSumsSingle4T<<<g, BLOCK_SIZE>>>((uint2*)scanned.GetPointer(), sums.GetConstPointer(), elems, scannedElemsPerBlock);
            }

        }
    }
    cudaDeviceSynchronize();
    timer.Stop();

    if(check)
    {
        nutty::Copy(h_devPrefixSum.Begin(), scanned.Begin(), scanned.End());
        if(mine)
        {
//             for(int j = 0; j < nutty::cuda::GetCudaGrid(elementCount, scannedElemsPerBlock); ++j)
//             {
//                 std::stringstream ss;
//                 ss << "j=" << j << "-" <<sums[j] << " ,";
//                 OutputDebugStringA(ss.str().c_str());
//             }
//             OutputDebugStringA("\n");
            for(int i = 0; i < data.Size(); ++i)
            {
                if(h_prefixSum[i] != h_devPrefixSum[i])
                {
                    std::stringstream ss;
                    ss <<"i=" << i << ", elem=" << (int)data[i] << ", " << (int)h_prefixSum[i] << "!=" << (int)h_devPrefixSum[i] << "\n";
                    OutputDebugStringA(ss.str().c_str());
                }
            }
        }
    }

    g_times.insert(std::pair<uint, double>(elementCount, timer.GetMillis() / (double)times));
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
    setlocale(LC_ALL, "German_Germany");
    std::ofstream scanProfile("scanProfile.txt");

    scanProfile.imbue(std::locale( "" ));

    int startBits = 24;
    int endBits = 25;

    for(int i = startBits; i < endBits; ++i)
    {
        int elementCount = 1 << i;
        scanProfile << elementCount << "\n";
    }
    scanProfile << "\n";

    for(int p = 256; p < 8*65536; p <<=1)
    {
        scanProfile << p << "\n";
    }

    for(int p = 256; p < 8*65536; p <<=1)
    {
        //scanProfile << "\n" << p << "\n";
        for(int i = startBits; i < endBits; ++i)
        {
            int elementCount = (1 << i);
            printf("Running test... n=%d\n", elementCount);
            test<true, true>(50 / 100.0f, elementCount, p);
            //scanProfile << elementCount << "\n";
        }

        for(auto it = g_times.begin(); it != g_times.end(); ++it)
        {
            scanProfile << it->second << "\n";
        }

        g_times.clear();
    }

//     scanProfile << "\n thrust \n";
// 
//     for(int i = 8; i < bits; ++i)
//     {
//         int elementCount = 1 << i;
//         test<false>(0.5f, elementCount);
//         scanProfile << elementCount << "\n";
//     }
// 
//     for(auto it = g_times.begin(); it != g_times.end(); ++it)
//     {
//         scanProfile << it->second << "\n";
//     }

    scanProfile << "error" << cudaDeviceSynchronize() << "\n";

    scanProfile.close();

    nutty::Release();

    return 0;
}