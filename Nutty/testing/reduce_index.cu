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

uint elementCount = 540;
int i = 0;

int randi(void)
{
    return elementCount - (i++); //(rand()%10);
}

int main(void)
{
#if 1
    _CrtSetDbgFlag (_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
#endif 

    nutty::Init();

    nutty::DeviceBuffer<uint> index(elementCount);
    nutty::DeviceBuffer<int> dataBuffer(elementCount);
    nutty::DeviceBuffer<int> result(elementCount);

    nutty::Fill(index.Begin(), index.End(), nutty::unary::Sequence<uint>());
    nutty::Fill(dataBuffer.Begin(), dataBuffer.End(), randi);

    nutty::ForEach(index.Begin(), index.End(), print);
    OutputDebugStringA("\n");
    nutty::ForEach(dataBuffer.Begin(), dataBuffer.End(), print);
    OutputDebugStringA("\n");

    nutty::base::ReduceIndexed(result.Begin(), dataBuffer.Begin(), dataBuffer.Begin() + elementCount, index.Begin(), 10, nutty::binary::Max<int>());

    nutty::ForEach(result.Begin(), result.End(), print);

    nutty::Release();

    return 0;
}