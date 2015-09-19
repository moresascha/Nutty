#pragma once
#include "../cuda/reduce.h"
#include "../Copy.h"
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    namespace base
    {

        template <
            typename IteratorDst,
            typename IteratorSrc,
            typename BinaryOperation,
            typename T
        >
        __host__ void Reduce1(
        IteratorDst& dst, 
        IteratorSrc& src,
        size_t d,
        BinaryOperation op,
        T neutral,
        cudaStream_t pStream = NULL)
        {
            uint elementsPerBlock = 256;

            //assert(d > 1);

            //todo: reduce small sets on the cpu

            nutty::cuda::Reduce(dst(), src(), neutral, d, op, elementsPerBlock, 0, pStream);

            if(d < elementsPerBlock)
            {
                return;
            }

            uint rest = (d % elementsPerBlock);

            uint grid = (uint)d / elementsPerBlock;

            if(rest > 0)
            {
                nutty::cuda::Reduce(dst() + grid, src(), neutral, rest, op, rest, elementsPerBlock * grid, pStream);
            }

            UINT elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);

            if(elementsLeft > 1)
            {
                base::Reduce1(dst, dst, elementsLeft, op, neutral, pStream);
            }
        }

        template <
            typename IteratorDst,
            typename IteratorSrc,
            typename BinaryOperation,
            typename T
        >
        __host__ void Reduce(
        IteratorDst& dst, 
        IteratorSrc& src,
        size_t d,
        BinaryOperation op,
        T neutral,
        cudaStream_t pStream = NULL)
        {
            const uint elementsPerBlock = 2*4096;//d / blockCount; //4 * 512;
            const uint blockSize = 256;//elementsPerBlock / 2;

            //assert(d > 1);
            uint grid = nutty::cuda::GetCudaGrid((uint)d, elementsPerBlock);
            if(grid > 1)//d > elementsPerBlock)
            {
                nutty::cuda::blockReduce<blockSize><<<grid, blockSize, 0, pStream>>>(src(), dst(), op, neutral, elementsPerBlock, (uint)d);
                nutty::cuda::blockReduce<blockSize><<<1, blockSize, 0, pStream>>>(dst(), dst(), op, neutral, grid, grid);
            }
            else
            {
                nutty::cuda::blockReduce<blockSize><<<1, blockSize, 0, pStream>>>(src(), dst(), op, neutral, d, (uint)d);
            }
        }

        template <
            typename IteratorDst,
            typename IteratorSrc,
            typename BinaryOperation,
            typename T
        >
        __device__ void ReduceDP(
        IteratorDst& dst, 
        IteratorSrc& src,
        size_t d,
        BinaryOperation op,
        T neutral,
        cudaStream_t stream = NULL)
        {
            uint elementsPerBlock = 512;

            //assert(d > 1);

            //todo: reduce small sets on the cpu
//             uint elementsLeft = 2;
// 
//             while(d > 0)
//             {
//                 nutty::cuda::ReduceDP(dst(), src(), neutral, d, op, elementsPerBlock, 0, stream);
// 
//                 if(d < elementsPerBlock)
//                 {
//                     return;
//                 }
// 
//                 uint rest = (d % elementsPerBlock);
// 
//                 uint grid = (uint)d / elementsPerBlock;
// 
//                 if(rest > 0)
//                 {
//                     nutty::cuda::ReduceDP(dst() + grid, src(), neutral, rest, op, rest, elementsPerBlock * grid, stream);
//                 }
// 
//                 src = dst;
//                 elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);
//                 d = elementsLeft;
//             }

            nutty::cuda::ReduceDP(dst(), src(), neutral, d, op, elementsPerBlock);

            if(d < elementsPerBlock)
            {
                return;
            }
 
            uint rest = (d % elementsPerBlock);

            uint grid = (uint)d / elementsPerBlock;

            if(rest > 0)
            {
                nutty::cuda::ReduceDP(dst() + grid, src(), neutral, rest, op, rest, elementsPerBlock * grid);
            }
  
            uint elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);

            if(elementsLeft > 1)
            {
                base::ReduceDP(dst, dst, elementsLeft, op, neutral);
            }
        }

        template <
            typename T,
            typename IteratorDst,
            typename IteratorSrc,
            typename IteratorIndex,
            typename BinaryOperation
        >
        void ReduceIndexed(
        IteratorDst dstBegin, 
        IteratorSrc srcBegin, 
        IteratorSrc srcEnd, 
        IteratorIndex indexBegin,
        T extreme,
        BinaryOperation op)
        {
            size_t d = Distance(srcBegin, srcEnd);

            uint elementsPerBlock = 512;

            assert(d > 1);

            //todo: reduce small sets on the cpu

            nutty::cuda::ReduceIndexed(dstBegin(), srcBegin(), d, indexBegin(), (uint)-1, extreme, op, elementsPerBlock);

            if(d < elementsPerBlock)
            {
                return;
            }

            uint rest = (d % elementsPerBlock);

            uint grid = d / elementsPerBlock;

            if(rest > 0)
            {
                nutty::cuda::ReduceIndexed(dstBegin() + grid, srcBegin(), rest, indexBegin(), (uint)-1, extreme, op, rest, elementsPerBlock * grid);
            } 

            UINT elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);

            if(elementsLeft > 1)
            {
                ReduceIndexed(dstBegin, srcBegin + elementsLeft, srcEnd, indexBegin, extreme, op);
            }
        }
    }
}