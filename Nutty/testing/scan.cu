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

std::stringstream g_ss;
void print(const int& t)
{
	g_ss.str("");
    g_ss << t;
    g_ss << " ";
    OutputDebugStringA(g_ss.str().c_str());
}

struct scandata
{
	int operator()()
	{
		int r = rand();
		return r/(float)RAND_MAX > 0 ? r : -1;
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
	size_t elems = 13;
	nutty::DeviceBuffer<int> toScan(elems, 0);

	nutty::Fill(toScan.Begin(), toScan.End(), scandata());

	nutty::ForEach(toScan.Begin(), toScan.End(), print);
	
	nutty::Scan(toScan.Begin(), toScan.End());
	OutputDebugStringA("\n");

	nutty::ForEach(toScan.Begin(), toScan.End(), print);
    
    //release nutty
    nutty::Release();

    return 0;
}