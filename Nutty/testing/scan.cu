#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Scan.h"
#include "../Functions.h"

std::stringstream g_ss; //te

void print(const int& t)
{
    g_ss.str("");
    g_ss << t;
    g_ss << " ";
    OutputDebugStringA(g_ss.str().c_str());
}

struct scandata
{
    int c;
    scandata(void) : c(0)
    {

    }
    int operator()()
    {
        int r = 1;//rand();
        if(c > 256)
        {
            r = 0;
        }
        c++;
        return 1;// % 2;
    }
};

struct Ray
{
    int a;
    int b;
};

void printRay(const Ray& r)
{
    OutputDebugStringA("Ray:(");
    print(r.a);
    print(r.b);
    OutputDebugStringA(") ");
}

struct RayData
{
    Ray operator()()
    {
        Ray r;
        r.a = rand() % 10;
        r.b = rand() % 10;
        return r;
    }
};

int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif

    //create nutty
    nutty::Init(); 

   // while(1)
    {
        srand(NULL);
        size_t size = 512*10; 
        nutty::DeviceBuffer<Ray> rays(size);
        nutty::DeviceBuffer<Ray> compactedRays(size);
        nutty::Fill(rays.Begin(), rays.End(), RayData());

        nutty::DeviceBuffer<int> mask(size, 0);
        nutty::Fill(mask.Begin(), mask.End(), scandata());

//         nutty::ForEach(mask.Begin(), mask.End(), print);
//         OutputDebugStringA("\n");

        nutty::DeviceBuffer<int> scannedMask(size, 0); 
        nutty::DeviceBuffer<int> sums(max(1, (int)(size/512)), 0);

        /*nutty::ForEach(rays.Begin(), rays.End(), printRay);
        OutputDebugStringA("\n");*/

        nutty::ExclusivePrefixSumScan(mask.Begin(), mask.End(), scannedMask.Begin(), sums.Begin());

        nutty::Compact(compactedRays.Begin(), rays.Begin(), rays.End(), mask.Begin(), scannedMask.Begin(), 0);

        /*nutty::ForEach(scannedMask.Begin(), scannedMask.End(), print);
        OutputDebugStringA("\n");*/
        nutty::ForEach(sums.Begin(), sums.End(), print);
        OutputDebugStringA("\n");
        /*
        nutty::ForEach(rays.Begin(), rays.End(), printRay);
        OutputDebugStringA("\n");}*/
        int t = *(scannedMask.End()-1);
        
        print(t);
        OutputDebugStringA("\n");
    }



    //release nutty
    nutty::Release();

    return 0;
}