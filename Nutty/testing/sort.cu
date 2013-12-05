#include <windows.h>
#include "../Nutty.h"
#include "../Fill.h"
#include "../Sort.h"
#include <sstream>
#include <fstream>
#include "../Inc.h"
#include "../ForEach.h"
#include "../Functions.h"
#include <ctime>

uint c = 0;

void print(const int& t)
{
    std::stringstream ss;
    ss << t;
    ss << " "; 
    OutputDebugStringA(ss.str().c_str());
   // if((++c % 16) == 0) OutputDebugStringA("\n");
}

struct sortinfo
{
    uint pos;
    int a;
    int b;
};

template < typename IT>
bool checkSort(IT& b, uint size, sortinfo* si)
{
    nutty::HostBuffer<int> cpy(b.Size());
    nutty::Copy(cpy.Begin(), b.Begin(), size);
    auto it = cpy.Begin();
    int i = 0;
    int _cc = 0;
    while(it != cpy.End())
    {
        int cc = *it;
        if(cc < i)
        {
            //si->pos = _cc;
            //si->a = i;
            //si->b = cc;
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
    
    uint end = 1 << 24;
    nutty::DeviceBuffer<int> a(end);

    srand(13123); //13123

    //nutty::ForEach(a.Begin(), a.End(), print);
    std::ofstream profileSort("sortProfile.txt");
    
   for(int i = 2; i <= end; ++i)
    {
        //int i = 2773;
        nutty::Fill(a.Begin(), a.Begin() + i - 1, rand);
        a.Insert(i - 1, -1);
        //nutty::ForEach(a.Begin(), a.Begin() + i, print);
        clock_t start = clock();
        nutty::Sort(a.Begin(), a.Begin() + i, nutty::BinaryDescending<int>());
        cudaDeviceSynchronize();
        int n = a[0];
        clock_t end = clock();
        double millis = (double)(end - start)/CLOCKS_PER_SEC;
        if(n != -1)
        {
            profileSort << n;
            profileSort << "\n\n\nError";
            profileSort.close();
            return 0;
        }
        profileSort << i;
        profileSort << " ";
        profileSort << millis;
        profileSort << "\n";
    }
    profileSort.close();
    nutty::Release();

    return 0;
}