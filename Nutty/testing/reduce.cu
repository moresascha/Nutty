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

    std::ifstream s("count.txt");

    uint count;
    s >> count;

    std::stringstream ss;
    ss << "count=" << count << "\n";
    OutputDebugStringA(ss.str().c_str());
    //create device memory (1.048.576 elements)
    
    //fill device memory with random numbers
    for(int i = 17; i <= 1 << 20; ++i)
    {
        ss.str("");
        ss << i << "\n";
        OutputDebugStringA(ss.str().c_str());
        nutty::DeviceBuffer<int> a(i);
        nutty::Fill(a.Begin(), a.End(), nutty::unary::Sequence<int>());
        nutty::Reduce(a.Begin(), a.End(), nutty::binary::Max<int>());
        
        uint r = a[0];
        if(r != i-1)
        {
            ss.str("");
            ss << (i-1);
            ss << " != ";
            ss << r << "\n";
            OutputDebugStringA(ss.str().c_str());
            nutty::ForEach(a.Begin(), a.End(), print);
            break;
        }
    }
    
    //release nutty
    nutty::Release();

    return 0;
}