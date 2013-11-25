#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"

namespace nutty
{
    namespace cuda
    {
        template <
            class T, T neutral
        >
        struct CompactScanOp
        {
            __device__ T operator()(T elem)
            {
                return elem == neutral ? 0 : 1;
            }

            T GetNeutral(void)
            {
                return neutral;
            }
        };

        template <
            typename T
        >
        struct PrefixSumOp
        {
            __device__ T operator()(T elem)
            {
                return elem;
            }

            T GetNeutral(void)
            {
                return 0;
            }
        };

        template <
            typename T,
            typename ST,
            typename Oprator
        >
        __global__ void _scan(Oprator op, T* content, ST* scanned, uint* sums, T neutralItem, uint startStage, uint length, uint writeSums)
        {
            ShrdMemory<T> s_mem;
            T* shrdMem = s_mem.Ptr();

            uint thid = threadIdx.x;
            uint grpId = blockIdx.x;
            uint globalPos = 2 * (thid + grpId * blockDim.x);

            uint gpos0 = globalPos;
            uint gpos1 = globalPos + 1;

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

            shrdMem[2 * (blockDim.x - thid - 1) + 0] = op(i1); //i1 == neutralItem ? 0 : (neutralItem==-2 ? i1 : 1);
            shrdMem[2 * (blockDim.x - thid - 1) + 1] = op(i0); //i0 == neutralItem ? 0 : (neutralItem==-2 ? i0 : 1);

            T last = op(i1); // == neutralItem ? 0 : 1;

            uint offset = 1;
            for(int i = startStage; i > 1; i >>= 1)
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
            }

            __syncthreads();

            for(int i = 1; i <= startStage; i <<= 1)
            {
                if(thid < i)
                {
                    int ai = offset*(2*thid);  
                    int bi = offset*(2*thid+1); 
                    if(bi < 2 * blockDim.x)
                    {
                        int t = shrdMem[ai];  
                        shrdMem[ai] += shrdMem[bi]; 
                        shrdMem[bi] = t;
                    }
                }
                offset >>= 1;
                __syncthreads();
            }

            int elemsLeft = 2 * blockDim.x ;

            if(gpos0 < length)
            {
                scanned[gpos0] = shrdMem[elemsLeft - 1 - (2*thid)];
            }

            if(gpos1 < length)
            {
                scanned[gpos1] = shrdMem[elemsLeft - 1 - (2*thid+1)];
            }

            if(thid == blockDim.x-1 && writeSums)
            {
                T _t = shrdMem[0];
                sums[blockIdx.x] = _t + last;
            }
        }

        template <
            typename T,
            typename ST
        >
        __device__ void _compact(T* content, ST* scanned, T neutral, uint N)
        {
            uint id = blockIdx.x * blockDim.x + threadIdx.x;

            if(id >= N)
            {
                return;
            }

            T t = scanned[id];

            if(t != neutral)
            {
                content[t] = content[id];
            }
        }

        template <
            typename T,
            typename ST
        >
        __global__ void compact(T* content, ST* scanned, T neutral, uint N)
        {
            _compact(content, scanned, neutral, N);
        }

        template <
            typename T,
            typename ST
        >
        __global__ void spreadSumsNCompact(T* content, ST* scanned, uint* scannedSums, T neutralElement, uint length)
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

            if(c != neutralElement)
            {
                int scanPos = scanned[gid] + scannedSums[grpId];
                content[scanPos] = c;
            }
        }

        const static size_t ELEMS_PER_BLOCK = 512;

        template <
            typename T
        >
        void PrefixSumScan(T* begin, T* end, T* prefixSum, uint* sums, size_t d)
        {
            PrefixSumOp<T> op0;
            PrefixSumOp<uint> op1;
            
            dim3 grid = cuda::getCudaGrid(d, ELEMS_PER_BLOCK);

            if(grid.x == 1)
            {
                Scan(begin, end, prefixSum, sums, NULL, d, op0, op1);
                return;
            }

            nutty::DeviceBuffer<uint> scannedSums(grid.x, op0.GetNeutral());

            Scan(begin, end, prefixSum, sums, scannedSums.Begin()(), d, op0, op1);

            spreadSumsNCompact<<<grid, ELEMS_PER_BLOCK>>>(begin, prefixSum, scannedSums.Begin()(), op0.GetNeutral(), d);
        }

        template <
            typename T
        >
        void CompactScan(T* begin, T* end, T* scanned, uint* sums, size_t d)
        {
            CompactScanOp<T, -1> op0;
            CompactScanOp<uint, -2> op1;

            dim3 grid = cuda::getCudaGrid(d, ELEMS_PER_BLOCK);

            if(grid.x == 1)
            {
                Scan(begin, end, scanned, sums, NULL, d, op0, op1);
                compact<<<1, d>>>(begin, scanned, op0.GetNeutral(), d);
                return;
            }

            nutty::DeviceBuffer<uint> scannedSums(grid.x, op0.GetNeutral());
            
            Scan(begin, end, scanned, sums, scannedSums.Begin()(), d, op0, op1);

            spreadSumsNCompact<<<grid, ELEMS_PER_BLOCK>>>(begin, scanned, scannedSums.Begin()(), op0.GetNeutral(), d);
        }

        template <
            typename T,
            typename OperatorFirstStage,
            typename OperatorSecondStage
        >
        void Scan(T* begin, T* end, T* scanned, uint* sums, uint* scannedSums, size_t d, OperatorFirstStage op0, OperatorSecondStage op1)
        {
            size_t elementsPerBlock = ELEMS_PER_BLOCK;

            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d % 2);
            }

            dim3 block = elementsPerBlock / 2;
            dim3 grid = cuda::getCudaGrid(d, elementsPerBlock);

            size_t startStage = elementsPerBlock / 2;

            if(!ispow2(startStage))
            {
                startStage = 1 << (getmsb(d));
            }

            _scan<<<grid, block, elementsPerBlock * sizeof(T)>>>(op0, begin, scanned, sums, op0.GetNeutral(), startStage, d, 1);

            if(grid.x == 1)
            {
                return;
            }

            dim3 sumBlock = (grid.x + (grid.x % 2)) / 2;
            _scan<<<1, sumBlock, 2 * sumBlock.x * sizeof(T)>>>(op1, sums, scannedSums, sums, op1.GetNeutral(), 4, grid.x, 0);
        }
    }
}