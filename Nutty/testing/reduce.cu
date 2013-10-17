#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#include "../Inc.h"
#include "../ForEach.h"
#include <time.h>

void print(const int& t)
{
    std::stringstream ss;
    ss << t;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
}

__device__ int& reduce_min(int& t0, int& t1)
{
    return t0 < t1 ? t0 : t1; 
}

int main(void)
{
    #define _CRTDBG_MAP_ALLOC

#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif 

    nutty::Init();

    srand((uint)time(NULL));

    nutty::DeviceBuffer<int> a(128);

    nutty::HostBuffer<int> r(1, 10);

    nutty::Fill(a.Begin(), a.End(), rand);

    //nutty::ForEach(a.Begin(), a.End(), print);

    //OutputDebugStringA("\n");

    print(r[0]);

    OutputDebugStringA("\n");

    nutty::Reduce(r, a, reduce_min);

    print(r[0]);

    //nutty::ForEach(a.Begin(), a.End(), print);

    nutty::Destroy();

    return 0;
}