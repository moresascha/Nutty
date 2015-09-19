#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"

namespace nutty
{
    namespace cuda
    {
        template <
            uint blockSize, 
            typename T,
            typename Operator
        >
        __global__ void blockReduce(const T* __restrict data, T* dstData, Operator op, T neutral, uint elemsPerBlock, uint N)
        {
            __shared__ T sdata[blockSize];

            uint tid = threadIdx.x;
            uint globalOffset = blockIdx.x * elemsPerBlock;
            uint i = threadIdx.x;

            T accu = neutral;

            while(i < elemsPerBlock && globalOffset + i < N) 
            { 
                accu = op(accu, op(data[globalOffset + i], globalOffset + i + blockSize < N ? data[globalOffset + i + blockSize] : neutral));
                i += 2 * blockSize;
            }

            sdata[threadIdx.x] = accu;

            __syncthreads();

            if(blockSize >= 512) { if(tid < 256) { sdata[tid] = op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
            if(blockSize >= 256) { if(tid < 128) { sdata[tid] = op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
            if(blockSize >= 128) { if(tid <  64)  { sdata[tid] = op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }

            if(tid < 32) 
            {
                if (blockSize >= 64) sdata[tid] = op(sdata[tid], sdata[tid + 32]);
                __syncthreads();

                if (blockSize >= 32) sdata[tid] = op(sdata[tid], sdata[tid + 16]);
                __syncthreads();

                if (blockSize >= 16) sdata[tid] = op(sdata[tid], sdata[tid + 8]);
                __syncthreads();

                if (blockSize >= 8) sdata[tid] = op(sdata[tid], sdata[tid + 4]);
                __syncthreads();

                if (blockSize >= 4) sdata[tid] = op(sdata[tid], sdata[tid + 2]);
                __syncthreads();

                if (blockSize >= 2) sdata[tid] = op(sdata[tid], sdata[tid + 1]);
                __syncthreads();
            }

            if(tid == 0) dstData[blockIdx.x] = sdata[0];
        }

//         template <
//             uint blockSize, 
//             typename T,
//             typename Operator
//         >
//         __global__ void oneGroupReduceAll(const T* __restrict data, T* dstData, Operator op, T neutral, uint N)
//         {
//             __shared__ T sdata[blockSize];
//             
//             uint tid = threadIdx.x;
//             uint i = threadIdx.x;
// 
//             T accu = neutral;
// 
//             while(i < N) 
//             { 
//                 accu = op(accu, op(data[i], i + blockSize < N ? data[i + blockSize] : neutral));
//                 i += 2 * blockSize;
//             }
// 
//             sdata[threadIdx.x] = accu;
// 
//             __syncthreads();
// 
//             if(blockSize >= 512) { if(tid < 256) { sdata[tid] = op(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
//             if(blockSize >= 256) { if(tid < 128) { sdata[tid] = op(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
//             if(blockSize >= 128) { if(tid <  64)  { sdata[tid] = op(sdata[tid], sdata[tid + 64]); } __syncthreads(); }
// 
//             //todo sync weg?
//             if(tid < 32) 
//             {
//                 if (blockSize >= 64) sdata[tid] = op(sdata[tid], sdata[tid + 32]);
//                 __syncthreads();
// 
//                 if (blockSize >= 32) sdata[tid] = op(sdata[tid], sdata[tid + 16]);
//                 __syncthreads();
// 
//                 if (blockSize >= 16) sdata[tid] = op(sdata[tid], sdata[tid + 8]);
//                 __syncthreads();
// 
//                 if (blockSize >= 8) sdata[tid] = op(sdata[tid], sdata[tid + 4]);
//                 __syncthreads();
// 
//                 if (blockSize >= 4) sdata[tid] = op(sdata[tid], sdata[tid + 2]);
//                 __syncthreads();
// 
//                 if (blockSize >= 2) sdata[tid] = op(sdata[tid], sdata[tid + 1]);
//                 __syncthreads();
//             }
// 
//             if(tid == 0) dstData[0] = sdata[0];
//         }

        template <
            typename _SM_T, 
            typename _SRC_T, 
            typename _DST_T, 
            typename _OPERATOR
        >
        __device__ void __reduce(_SM_T s_d, _SRC_T d0, _SRC_T d1, _DST_T dst, uint startStage, _OPERATOR _operator)
        {
            uint id = threadIdx.x;

            s_d[id] = _operator(d0, d1);

            __syncthreads();

            for(int i = startStage; i > 0; i >>= 1)
            {
                if(id < i && id + i < blockDim.x)
                {
                    s_d[id] = _operator(s_d[id + i], s_d[id]);
                }
                
                __syncthreads();
            }

            if(id == 0)
            {
                dst[blockIdx.x] = s_d[0];
            } 
        }

        template< 
            typename T_SRC, 
            typename T_DST,
            typename BinaryOperator
        >
        __global__ void reduce(T_SRC* data, T_DST* dst, BinaryOperator _operator, uint stride, uint startStage, uint length, T_SRC neutral, uint memoryOffset)
        {
            uint si = blockIdx.x * stride + threadIdx.x;

            T_SRC d0 = data[memoryOffset + si];
            T_SRC d1 = neutral;

            if(memoryOffset + si + blockDim.x < length)
            {
                d1 = data[memoryOffset + si + blockDim.x];
            }

            ShrdMemory<T_DST> shrd;

            __reduce(shrd.Ptr(), d0, d1, dst, startStage, _operator);
        }

        template< 
            typename T_SRC, 
            typename T_DST,
            typename T_ID,
            typename BinaryOperator
        >
        __global__ void reduceIndexed(T_SRC* data, T_DST* dst, BinaryOperator _operator, T_ID* index, 
                                      T_SRC extremes, uint invalidIndex, uint stride, uint startStage, uint length, uint memoryOffset)
        {
            ShrdMemory<T_DST> shrd;

            uint si = blockIdx.x * stride + threadIdx.x;

            T_ID i0 = index[memoryOffset + si];
            T_ID i1 = i0;

            if(memoryOffset + si + blockDim.x < length)
            {
                i1 = index[memoryOffset + si + blockDim.x];
            }

            T_SRC d0;
            T_SRC d1;

            if(i0 == invalidIndex)
            {
                d0 = extremes;
            }
            else
            {
                d0 = data[i0];
            }

            if(i1 == invalidIndex)
            {
                d1 = extremes;
            }
            else
            {
                d1 = data[i1];
            }

            __reduce(shrd.Ptr(), d0, d1, dst, startStage, _operator);
        }

        template <
            typename T,
            typename T_ID,
            typename BinaryOperation
        >
        void ReduceIndexed(T* dst, T* src, size_t d, T_ID* index, T_ID invalidIndex, T extreme, BinaryOperation op, uint elementsPerBlock, uint memoryOffset = 0)
        {
            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d%2);
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = max(1, (uint)((d / 2) / block.x));

            uint startStage = elementsPerBlock;

            if(!Ispow2(startStage))
            {
                startStage = 1 << (GetMSB(d)+1);
            }

            reduceIndexed
                <<<grid, block, block.x * sizeof(T), 0>>>
                (src, dst, op, index, extreme, invalidIndex, elementsPerBlock, startStage, memoryOffset + (uint)d, memoryOffset);
        }

//         template <
//             uint blockSize,
//             typename T,
//             typename BinaryOperation
//         >
//         __host__ void ReducePerBlock(T* dst, T* src, T neutral, size_t d, BinaryOperation op, uint elementsPerBlock, cudaStream_t pStream = NULL)
//         {
//             uint grid = nutty::cuda::GetCudaGrid((uint)d, elementsPerBlock);
//             blockReduce<blockSize><<<grid, blockSize, 0, pStream>>>(src, dst, op, neutral, (uint)d);
//         }

        template <
            typename T,
            typename BinaryOperation
        >
        __host__ void Reduce(T* dst, T* src, T neutral, size_t d, BinaryOperation op, uint elementsPerBlock, uint memoryOffset = 0, cudaStream_t pStream = NULL)
        {
            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d%2);
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = max(1, (uint)((d / 2) / block.x));

            uint startStage = elementsPerBlock;

            if(!Ispow2(startStage))
            {
                startStage = (1 << (GetMSB((uint)d)+1));
            }

            reduce
                <<<grid, block, block.x * sizeof(T), pStream>>>
                (src, dst, op, elementsPerBlock, startStage, memoryOffset + (uint)d, neutral, memoryOffset);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        __device__ void ReduceDP(T* dst, T* src, T neutral, size_t d, BinaryOperation op, uint elementsPerBlock, uint memoryOffset = 0, cudaStream_t stream = 0)
        {
            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d%2);
            }

            dim3 block = elementsPerBlock / 2; 
            dim3 grid = max(1, (uint)((d / 2) / block.x));

            uint startStage = elementsPerBlock;

            if(!Ispow2(startStage))
            {
                startStage = (1 << (GetMSB((uint)d)+1));
            }

            reduce
                <<<grid, block, block.x * sizeof(T), stream>>>
                (src, dst, op, elementsPerBlock, startStage, memoryOffset + (uint)d, neutral, memoryOffset);
        }
    }
}