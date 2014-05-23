#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"

namespace nutty
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

    namespace cuda
    {
        template <
            typename T,
            typename I,
            typename Operator,
            int WRITE_SUM,
            int IS_INCLUSIVE
        >
        __global__ void _scan(Operator op, const T* content, I* scanned, I* sums, T neutralItem, uint startStage, uint length)
        {
            ShrdMemory<I> s_mem;
            I* shrdMem = s_mem.Ptr();

            uint thid = threadIdx.x;
            uint grpId = blockIdx.x;
            uint globalPos = 2 * (thid + grpId * blockDim.x);

            uint gpos0 = globalPos;
            uint gpos1 = globalPos + 1;

            T i0 = neutralItem;
            T i1 = neutralItem;

            if(gpos0 < length)
            {
                i0 = content[gpos0];
            }

            if(gpos1 < length)
            {
                i1 = content[gpos1];
            }

            //todo
            /*uint ai = thid;
            uint bi = thid + (n/2);

            uint bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            uint bankOffsetB = CONFLICT_FREE_OFFSET(bi);*/

            shrdMem[2 * (blockDim.x - thid - 1) + 0] = (I)op(i1); //i1 == neutralItem ? 0 : (neutralItem==-2 ? i1 : 1);
            shrdMem[2 * (blockDim.x - thid - 1) + 1] = (I)op(i0); //i0 == neutralItem ? 0 : (neutralItem==-2 ? i0 : 1);

            T last = op(i1); // == neutralItem ? 0 : 1;

            uint offset = 1;

            __syncthreads();

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

            int elemsLeft = 2 * blockDim.x;

            __shared__ I tmp;

            if(IS_INCLUSIVE)
            {
                int dst1 = (2*thid) + 2;
                I s0 = shrdMem[(2*thid) + 0];
                I s1 = shrdMem[(2*thid) + 1];

                __syncthreads();

                if(dst1 < elemsLeft)
                {
                    shrdMem[dst1] = s1;
                }

                shrdMem[(2*thid) + 1] = s0;
                                
                __syncthreads();

                if(gpos1 == length-1 || gpos0 == length-1)
                {
                    tmp = gpos1 == length-1 ? op(i1) : op(i0);
                    shrdMem[0] += gpos1 == length-1 ? op(i1) : op(i0);
                }
                else if(thid == blockDim.x - 1)
                {
                    tmp = i1;
                    shrdMem[0] += op(i1);
                }

                __syncthreads();
            }

            if(gpos0 < length)
            {
                scanned[gpos0] = shrdMem[elemsLeft - 1 - (2*thid)];
            }

            if(gpos1 < length)
            {
                scanned[gpos1] = shrdMem[elemsLeft - 1 - (2*thid+1)];
            }

            if(WRITE_SUM && thid == blockDim.x-1)
            {
                I _t = shrdMem[0];
                if(IS_INCLUSIVE)
                {
                    sums[blockIdx.x] = _t - tmp;
                }
                else
                {
                    sums[blockIdx.x] = _t + last;
                }
            }
        }

        template <
            typename T,
            typename ST
        >
        __device__ void _compact(T* compactedContent, const T* content, const ST* mask, const ST* scanned, ST neutral, uint N)
        {
            uint id = blockIdx.x * blockDim.x + threadIdx.x;

            if(id >= N)
            {
                return;
            }

            ST t = mask[id];

            if(t != neutral)
            {
                t = scanned[id];
                compactedContent[t] = content[id];
            }
        }

        template <
            typename T,
            typename ST
        >
        __global__ void compact(T* dstContent, T* content, ST* mask, ST* scanned, ST neutral, uint N)
        {
            _compact(dstContent, content, mask, scanned, neutral, N);
        }

        template <
            typename T
        >
        __global__ void spreadSums(T* scanned, T* prefixSum, uint length)
        {
            uint thid = threadIdx.x;
            uint grpId = blockIdx.x;
            uint N = blockDim.x;
            uint gid = N + grpId * N + thid;

            if(gid >= length)
            {
                return;
            }

            scanned[gid] = scanned[gid] + prefixSum[grpId+1];
        }

//         template <
//             typename T,
//             int steps
//         >
//         __global__ void shiftMemory(T* memory, uint length)
//         {
//             uint id = threadIdx.x + blockIdx.x * blockDim.x;
// 
//             extern __shared__ T s_data[];
// 
//             if(id >= length)
//             {
//                 return;
//             }
// 
//             s_data[threadIdx.x] = memory[id];
// 
//             __syncthreads();
// 
//             if(steps < 0 && (int)threadIdx.x + steps <= 0)
//             {
//                 return;
//             }
// 
//             if(steps > 0 && (int)threadIdx.x + steps >= blockDim.x)
//             {
//                 return;
//             }
// 
//             memory[id + steps] = s_data[threadIdx.x];
//         }

        template <
            typename T,
            typename ST
        >
        void Compact(T* dst, T* src, ST* mask, ST* scanned, ST neutralElement, size_t d)
        {
            dim3 grid = cuda::GetCudaGrid(d, (size_t)256);
            compact<<<grid.x, 256, 0, 0>>>(dst, src, mask, scanned, neutralElement, d);
        }

        const static size_t ELEMS_PER_BLOCK = 512;

        template <
            typename T,
            typename I
        >
        void ExclusivePrefixSumScan(T* begin, I* prefixSum, I* sums, size_t d)
        {
           PrefixSumOp<T> op;
           _ExclusiveScan(begin, prefixSum, sums, d, op);
        }

        template <
            typename T,
            typename I,
            typename Operation,
            int TYPE
        >
        void __Scan(T* begin, I* scanned, I* sums, I* scannedSums, size_t d, Operation op)
        {
            size_t elementsPerBlock = ELEMS_PER_BLOCK;

            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d % 2);
            }

            dim3 block = elementsPerBlock / 2;
            dim3 grid = cuda::GetCudaGrid(d, elementsPerBlock);

            size_t startStage = elementsPerBlock / 2;

            if(!Ispow2(startStage))
            {
                startStage = 1ULL << (GetMSB(startStage) + 1);
            }

            _scan<T, I, Operation, 1, TYPE><<<grid, block, elementsPerBlock * sizeof(I), 0>>>(op, begin, scanned, sums, op.GetNeutral(), startStage, d);

            if(grid.x == 1)
            {
                return;
            }

            dim3 sumBlock = (grid.x + (grid.x % 2)) / 2;

            startStage = sumBlock.x;

            if(!Ispow2(startStage))
            {
                startStage = 1ULL << (GetMSB(startStage) + 1);
            }

            PrefixSumOp<I> _op;

            _scan<I, I, PrefixSumOp<I>, 0, 0><<<1, sumBlock, 2 * sumBlock.x * sizeof(I), 0>>>(_op, sums, scannedSums, (I*)NULL, _op.GetNeutral(), startStage, grid.x);
        }

        template <
            typename T,
            typename I,
            typename Operator,
            int TYPE
        >
        void _Scan(T* begin, I* prefixSum, I* sums, size_t d, Operator op)
        {            
            dim3 grid = cuda::GetCudaGrid(d, ELEMS_PER_BLOCK);

            if(grid.x == 1)
            {
                __Scan<T, I, Operator, TYPE>(begin, prefixSum, sums, (I*)NULL, d, op);
                return;
            }

            nutty::DeviceBuffer<I> scannedSums(grid.x, op.GetNeutral());

            __Scan<T, I, Operator, TYPE>(begin, prefixSum, sums, scannedSums.Begin()(), d, op);

            grid.x = grid.x - 1;

            assert(grid.x > 0);

            spreadSums<<<grid, ELEMS_PER_BLOCK, 0, 0>>>(prefixSum, scannedSums.Begin()(), d);
        }

        template <
            typename T,
            typename I,
            typename Operator
        >
        void _ExclusiveScan(T* begin, I* prefixSum, I* sums, size_t d, Operator op)
        {            
            _Scan<T, I, Operator, 0>(begin, prefixSum, sums, d, op);
        }

        template <
            typename T,
            typename I,
            typename Operator
        >
        void _InclusiveScan(T* begin, I* prefixSum, I* sums, size_t d, Operator op)
        {            
            _Scan<T, I, Operator, 1>(begin, prefixSum, sums, d, op);
        }
    }
}