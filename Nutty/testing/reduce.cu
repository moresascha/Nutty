#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Reduce.h"
#include <sstream>
#include <fstream>
#define _CRTDBG_MAP_ALLOC

#include "../Inc.h"
#include "../ForEach.h"
#include "../Functions.h"

void print(const int& t)
{
    std::stringstream ss;
    ss << t;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
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

    //create device memory (1.048.576 elements)
    nutty::DeviceBuffer<int> a(1 << 20);

    //fill device memory with random numbers
    nutty::Fill(a.Begin(), a.End(), nutty::unary::RandMax<int>(10));

    //set -1 at some random position
    a.Insert(a.Begin() + (1 << 16), -1);

    //parallel min reduction
    int min = nutty::Reduce(nutty::binary::Min<int>(), a);
    
    //check
    assert(min == -1);

    //release nutty
    nutty::Release();

    return 0;
}