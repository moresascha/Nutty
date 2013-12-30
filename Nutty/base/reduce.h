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
        void Reduce(
        IteratorDst& dst, 
        IteratorSrc& src,
        size_t d,
        BinaryOperation op,
        T neutral)
        {
            uint elementsPerBlock = 512;
            
            //assert(d > 1);

            //todo: reduce small sets on the cpu

            nutty::cuda::Reduce(dst(), src(), neutral, d, op, elementsPerBlock);

            if(d < elementsPerBlock)
            {
                return;
            }

            uint rest = (d % elementsPerBlock);

            uint grid = (uint)d / elementsPerBlock;

            if(rest > 0)
            {
                nutty::cuda::Reduce(dst() + grid, src(), neutral, rest, op, rest, elementsPerBlock * grid);
            }

            UINT elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);

            if(elementsLeft > 1)
            {
                base::Reduce(dst, dst, elementsLeft, op, neutral);
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