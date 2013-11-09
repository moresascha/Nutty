#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Sort.h"
#include <sstream>
#include <fstream>
#include "../Inc.h"
#include "../ForEach.h"
#include "../Functions.h"

uint c = 0;

void print(const int& t)
{
    std::stringstream ss;
    ss << t;
    ss << " ";
    OutputDebugStringA(ss.str().c_str());
    if((++c % 16) == 0) OutputDebugStringA("\n");
}

struct sortinfo
{
    uint pos;
    int a;
    int b;
};

bool checkSort(nutty::DeviceBuffer<int>& b, sortinfo* si)
{
    nutty::HostBuffer<int> cpy(b.Size());
    nutty::Copy(cpy.Begin(), b.Begin(), b.Size());
    auto it = cpy.Begin();
    int i = 0;
    int _cc = 0;
    while(it != cpy.End())
    {
        int cc = *it;
        if(cc < i)
        {
            si->pos = _cc;
            si->a = i;
            si->b = cc;
            /*std::stringstream m;
            m << cc;
            m << " ";
            m << i << ", ";
            m << _cc;
            OutputDebugStringA(m.str().c_str());
            OutputDebugStringA("\n");*/
            return false;
        }
        i = cc;
        it++;
        _cc++;
    }
    return true;
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
    sortinfo si;
    for(int i = 1; i < 2; ++i)
    {
        nutty::DeviceBuffer<int> a(16);
        nutty::DeviceBuffer<int> key(16);
        srand(13123); //13123

        nutty::Fill(a.Begin(), a.End(), rand);
        nutty::Fill(key.Begin(), key.End(), nutty::unary::Sequence<int>());
        
        nutty::ForEach(a.Begin(), a.End(), print);
        nutty::ForEach(key.Begin(), key.End(), print);
        /*a.Insert(0, 10124); 
        a.Insert(1, 25584);
        a.Insert(2, 30488);
        a.Insert(3, 87);
        a.Insert(4, 2407);
        a.Insert(5, 30977);
        a.Insert(6, 24132);
        a.Insert(7, 21753);
        a.Insert(8, 30033);
        a.Insert(9, 24335);
        a.Insert(10, 30960);
        a.Insert(11, 16301);
        a.Insert(12, 8520);
        a.Insert(13, 16950);
        a.Insert(14, 6301);
        a.Insert(15, 2695);
        a.Insert(16, 13148);
        a.Insert(17, 25424);
        a.Insert(18, 8582);
        a.Insert(19, 21598);
        a.Insert(20, 11353);
        a.Insert(21, 26639);*/
    
        //nutty::ForEach(a.Begin(), a.End(), print);

        //OutputDebugStringA("\n");
        c = 0;

        //nutty::SortDescending(a);
        nutty::Sort(key.Begin(), key.End(), a.Begin(), nutty::BinaryDescending<int>());
        nutty::ForEach(a.Begin(), a.End(), print);
        nutty::ForEach(key.Begin(), key.End(), print);
        //
        /*
        if(!checkSort(a, &si))
        {
            std::stringstream ss;
            ss << "N=" << i << ", pos=" << si.pos << ", values=" << si.a << " > " << si.b << "\n";
            OutputDebugStringA(ss.str().c_str());
            nutty::ForEach(a.Begin(), a.End(), print);
            break;
        }
        else
        {
            //nutty::ForEach(a.Begin(), a.End(), print);
        }*/

    }

    nutty::Release();

    return 0;
}