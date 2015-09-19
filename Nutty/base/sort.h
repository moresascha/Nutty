#pragma once
#include "../cuda/sort.h"

namespace nutty
{
    template <
        typename T
    >
    struct BinaryDescending
    {
        __device__ ___host char operator()(T f0, T f1)
        {
            return f0 > f1;
        }
    };

    template <
        typename T
    >
    struct BinaryAscending
    {
        __device__ ___host char operator()(T f0, T f1)
        {
            return f0 < f1;
        }
    };

    namespace base
    {
        template <
            typename Iterator_,
            typename BinaryOperation
        >
        ___host void SortPerGroup(
            Iterator_& begin, 
            uint elementsPerBlock, uint startStage, uint endStage, uint startStep, uint length, BinaryOperation op, cudaStream_t pStream = NULL)
        {
            nutty::cuda::SortPerGroup(begin, elementsPerBlock, startStage, endStage, startStep, length, op, pStream);
        }

        template<
            typename Iterator_,
            typename BinaryOperation
        >
        ___host void SortStep(Iterator_& start, uint grid, uint block, uint stage, uint step, uint length, BinaryOperation op, cudaStream_t pStream = NULL)
        {
            nutty::cuda::SortStep(start, grid, block, stage, step, length, op, pStream);
        }

        template <
            typename Iterator_,
            typename BinaryOperation
        >
        ___host void Sort(Iterator_& start, Iterator_& end, BinaryOperation op, cudaStream_t pStream = NULL)
        {
            size_t d = Distance(start, end);
            base::Sort(start, d, op, pStream);
        }

        template <
            typename Iterator_,
            typename BinaryOperation
        >
        ___host void Sort(Iterator_& start, size_t d, BinaryOperation op, cudaStream_t pStream = NULL)
        {
            uint length = (uint)d;

            uint elemCount = length;

            const uint maxElemsBlock = 512;

            uint elemsPerBlock = maxElemsBlock;

            if(elemCount >= elemsPerBlock)
            {
                elemCount = elemsPerBlock;
            }
            else
            {
                elemsPerBlock = elemCount;
            }

            uint perGroupEndStage = elemsPerBlock;
            if(elemsPerBlock & (elemsPerBlock-1))
            {
                perGroupEndStage = 1 << (GetMSB(elemsPerBlock) + 1);
            }

            SortPerGroup(start, elemsPerBlock, 2, perGroupEndStage, 1, length, op, pStream);

            elemCount = length;
            elemsPerBlock = maxElemsBlock;

            if(elemCount <= elemsPerBlock)
            {
                return;
            }

            uint stageStart = elemsPerBlock << 1;
            uint grid = cuda::GetCudaGrid(length, elemsPerBlock);

            uint endStage = length;
            if(length & (length-1))
            {
                endStage = 1 << (GetMSB(length) + 1);
            }

            for(uint pow2stage = stageStart; pow2stage <= endStage; pow2stage <<= 1)
            {
                for(uint step = pow2stage >> 1; step > 0; step = step >> 1)
                {
                    if((step << 1) <= elemsPerBlock)
                    {
                        SortPerGroup(start, elemsPerBlock, pow2stage, pow2stage, step, length, op, pStream);
                        break;
                    }
                    else
                    {
                        SortStep(start, grid, elemsPerBlock, pow2stage, step, length, op, pStream);
                    }
                }
            }
        }

        //key value
        template <
            typename K,
            typename T, 
            typename BinaryOperation
        >
        void SortKeyStep(
        Iterator<
                K, nutty::base::Base_Buffer<K, nutty::DeviceContent<K>, nutty::CudaAllocator<K>>
                >& keyBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& values, 
        uint grid, uint block, uint stage, uint step, BinaryOperation op, uint offset = 0)
        {
            nutty::cuda::SortKeyStep(keyBegin, values, grid, block, stage, step, op, offset);
        }

        template <
            typename IteratorKey,
            typename IteratorData,
            typename BinaryOperation
        >
        void Sort(IteratorKey& keyStart, IteratorKey& keyEnd, IteratorData& valuesBegin, BinaryOperation op, cudaStream_t pStream = NULL)
        {
            uint length = (uint)Distance(keyStart, keyEnd);

            uint elemCount = length;

            const uint maxElemsBlock = 512;

            uint elemsPerBlock = maxElemsBlock;

            if(elemCount >= elemsPerBlock)
            {
                elemCount = elemsPerBlock;
            }
            else
            {
                elemsPerBlock = elemCount;
            }

            uint perGroupEndStage = elemsPerBlock;
            if(elemsPerBlock & (elemsPerBlock-1))
            {
                perGroupEndStage = 1 << (GetMSB(elemsPerBlock) + 1);
            }

            nutty::cuda::SortKeyPerGroup(keyStart, keyEnd, valuesBegin, elemsPerBlock, 2, perGroupEndStage, 1, length, op, pStream);

            elemCount = length;
            elemsPerBlock = maxElemsBlock;

            if(elemCount <= elemsPerBlock)
            {
                return;
            }

            uint stageStart = elemsPerBlock << 1;
            uint grid = cuda::GetCudaGrid(length, elemsPerBlock);

            uint endStage = length;
            if(length & (length-1))
            {
                endStage = 1 << (GetMSB(length) + 1);
            }

            for(uint pow2stage = stageStart; pow2stage <= endStage; pow2stage <<= 1)
            {
                for(uint step = pow2stage >> 1; step > 0; step = step >> 1)
                {
                    if((step << 1) <= elemsPerBlock)
                    {
                        nutty::cuda::SortKeyPerGroup(keyStart, keyEnd, valuesBegin, elemsPerBlock, pow2stage, pow2stage, step, length, op, pStream);
                        break;
                    }
                    else
                    {
                        nutty::cuda::SortKeyStep(keyStart, valuesBegin, grid, elemsPerBlock, pow2stage, step, length, op, pStream);
                    }
                }
            }
        }
    }
}