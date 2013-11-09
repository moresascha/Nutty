#pragma once
#include "cuda/Globals.cuh"
#include "cuda/cuda_helper.h"
#include "base/Iterator.h"
#include "Inc.h"

namespace nutty
{
	template <
		typename T
	>
	__global__ void scan(T* content, int* scanned, int* sums, T neutralItem, uint startStage, uint length, uint writeSums, uint memoryOffset = 0)
	{
		ShrdMemory<T> s_mem;
		T* shrdMem = s_mem.Ptr();

		uint thid = threadIdx.x;
		uint grpId = blockIdx.x;
        uint globalPos = 2 * (thid + grpId * blockDim.x);

        uint gpos0 = memoryOffset + globalPos;
        uint gpos1 = memoryOffset + globalPos + 1;

		uint i0 = neutralItem;
		uint i1 = neutralItem;
  
        if(gpos0 < length)
        {
            i0 = content[gpos0];
        }

        if(gpos1 < length)
        {
            i1 = content[gpos1];
        }

		if(neutralItem == i0)
		{
			shrdMem[2 * thid] = 0;
		}
		else
		{
			shrdMem[2 * thid] = 1;
		}

		if(neutralItem == i1)
		{
			shrdMem[2 * thid + 1] = 0;
		}
		else
		{
			shrdMem[2 * thid + 1] = 1;
		}

		uint last = shrdMem[2*thid+1];

        uint offset = 1;
        for(int i = startStage; i > 0; i >>= 1)
        {
            if(thid < i)
            {
                int ai = offset*(2*thid);  
                int bi = offset*(2*thid+1);  
                if(bi < 2 * blockDim.x)
                {
                    shrdMem[ai] += shrdMem[bi];
                }
            }
            offset <<= 1;
            __syncthreads();
        }

        if(thid == 0) 
        { 
            shrdMem[0] = 0;
			
/*
            if(length < (grpId+1) * (startStage<<1))
            {
                //shrdMem[2*blockDim.x-1] += shrdMem[startStage - 1];

                int ai = startStage - 1;
                int bi = 2*blockDim.x-1;
   
                int t = shrdMem[ai];  
                shrdMem[ai] = shrdMem[bi]; 
                shrdMem[bi] += t;
            }
*/

        }

        __syncthreads();

        for(int i = 1; i <= startStage; i <<= 1)
        {
            offset >>= 1;

            if(thid < i)
            {
				int bi = offset*(2*thid);  
				int ai = offset*(2*thid+1); 
                if(bi < 2 * blockDim.x)
                {
                    int t = shrdMem[ai];  
                    shrdMem[ai] = shrdMem[bi]; 
                    shrdMem[bi] += t;
                }
            }
            __syncthreads();
        }

        if(gpos0 < length)
        {
            scanned[gpos0] = shrdMem[(2*thid)];
        }

        if(gpos1 < length)
        {
            scanned[gpos1] = shrdMem[(2*thid+1)];
        }

		if(thid == (blockDim.x-1) && writeSums)
		{
			sums[blockIdx.x] = shrdMem[0] + last;
		}
	}

	template <
		typename T
	>
	__global__ void spreadSums(T* content, uint* scanned, uint* sums, uint length)
	{
		uint thid = threadIdx.x;
		uint grpId = blockIdx.x;
		uint N = blockDim.x;
        uint gid = grpId * N + thid;

        if(gid >= length)
        {
            return;
        }

        T c = content[gid];

        if(c != -1)
        {
            int scanPos = scanned[gid] + (grpId > 0 ? sums[grpId-1] : 0);
            content[scanPos] = c;
        }
	}

	std::stringstream g_ss;
	void print(const int& t)
	{
		g_ss.str("");
		g_ss << t;
		g_ss << " ";
		OutputDebugStringA(g_ss.str().c_str());
	}

	template <
		typename T,
		typename C
	>
	void Scan(Iterator<T, C>& begin, Iterator<T, C>& end)
	{
        uint d = nutty::Distance(begin, end);
        uint elementsPerBlock = 512;

        if(elementsPerBlock >= d)
        {
            elementsPerBlock = (uint)d + (d%2);
        }

        dim3 block = elementsPerBlock / 2; 
        dim3 grid = cuda::getCudaGrid(d, elementsPerBlock);

        uint startStage = elementsPerBlock/2;

        if(!ispow2(startStage))
        {
            startStage = 1 << (getmsb(d));
        }

		nutty::DeviceBuffer<T> sums(grid.x);
        nutty::DeviceBuffer<T> scanned(d);

        scan<<<grid, block, elementsPerBlock * sizeof(T)>>>(begin(), scanned.Begin()(), sums.Begin()(), -1, startStage, d, 1);

        dim3 sumGrid = 1;
        dim3 sumBlock = 2;//(grid.x + (grid.x % 2)) / 2;
        //scan<<<sumGrid, sumBlock, 2 * sumBlock.x * sizeof(T)>>>(begin(), begin(), begin(), 0, 8, sums.Size(), 0);
        OutputDebugStringA("\n");
		nutty::ForEach(sums.Begin(), sums.End(), print);
        OutputDebugStringA("\n");
        nutty::ForEach(scanned.Begin(), scanned.End(), print);
        grid.x -= 1;
//		spreadSums<<<grid, elementsPerBlock>>>(begin(), scanned.Begin()(), sums.Begin()(), d);
	}
}