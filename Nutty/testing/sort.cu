#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Sort.h"
#include <sstream>
#include <fstream>
#include "../Inc.h"
#include "../ForEach.h"

void print(const int& t)
{
    std::stringstream ss;
    ss << t;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
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

    nutty::DeviceBuffer<int> a(128);

    nutty::Generate(a.Begin(), a.End(), rand);

    nutty::ForEach(a.Begin(), a.End(), print);

    OutputDebugStringA("\n");

    //nutty::Generate(b.Begin(), b.End(), rand);

    nutty::SortDescending(a);

    nutty::ForEach(a.Begin(), a.End(), print);

    nutty::Destroy();

    return 0;
}