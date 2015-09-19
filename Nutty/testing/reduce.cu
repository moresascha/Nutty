#include "test_help.cuh"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Functions.h"

#include "../cuTimer.h"


int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif 

    //create nutty
    nutty::Init();
    std::stringstream profileStream;
    profileStream.imbue(std::locale( "" ));

    int startBits = 24;
    int endBits = 25;

    for(int i = startBits; i < endBits; ++i)
    {
        int elementCount = 1 << i;
        profileStream << elementCount << "\n";
    }

    profileStream << "\n";
    for(uint grid = 1; grid < 24; grid+=1)
        {
                       uint elementsPerBlock = nutty::cuda::GetCudaGrid((uint)(1 << startBits), grid);
                profileStream << grid << "\n";
        }
    profileStream << "\n";
    for(uint grid = 1; grid < 24; grid+=1)
    {
        uint elementCount = 1 << startBits;
        nutty::DeviceBuffer<int> data(elementCount);

        nutty::Fill(data.Begin(), data.End(), nutty::unary::Sequence<int>());

       // for(int elemsPerBlock = elementCount; elemsPerBlock; elemsPerBlock/=2)
        {
            nutty::cuTimer timer;

            timer.Start();
            for(int s = 0; s < 32; ++s)
            {
                nutty::binary::Max<int> op;
//                 for(int k = 0; k < grid; ++k)
//                 nutty::Reduce(data.Begin(), data.End(), nutty::binary::Max<int>(), 0);

/*                const uint elementsPerBlock = 2*4096;//d / blockCount; //4 * 512;*/
                const uint blockSize = 256;//elementsPerBlock / 2;

                //assert(d > 1);
                //uint grid = nutty::cuda::GetCudaGrid((uint)elementCount, elementsPerBlock);
                uint elementsPerBlock = nutty::cuda::GetCudaGrid((uint)elementCount, grid);
                nutty::cuda::blockReduce<blockSize><<<grid, blockSize>>>(data.Begin()(), data.Begin()(), op, 0, elementsPerBlock, (uint)elementCount);
            }
                
            timer.Stop();
            profileStream << timer.GetMillis()/32.0 << "\n";
            if(elementCount-1 != data[0])
            {
                print(data[0]);
            }
        }
    }
    OutputDebugStringA("\n");
    OutputDebugStringA(profileStream.str().c_str());

    //release nutty
    nutty::Release();


    return 0;
}