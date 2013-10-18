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

    nutty::Init();

    nutty::DeviceBuffer<int> a(128);

    nutty::Fill(a.Begin(), a.End(), nutty::unary::RandMax<int>(10));

    nutty::ForEach(a.Begin(), a.End(), print);

    OutputDebugStringA("\n");

    a.Insert(a.Begin() + 65, -1);

    nutty::ForEach(a.Begin(), a.End(), print);

    OutputDebugStringA("\n");

    nutty::Reduce(a.Begin() + 64, a.End(), nutty::binary::Min<int>());
    
    print(a[64]);

    nutty::Release();

    return 0;
}