#pragma once
#include "../cuda/sort.h"
#include "../HostBuffer.h"
#include "../DeviceBuffer.h"

namespace nutty
{
    template <
        typename T
    >
    struct BinaryDescending
    {
        __device__ __host__ char operator()(T f0, T f1)
        {
            return f0 > f1;
        }
    };

    template <
        typename T
    >
    struct BinaryAscending
    {
        __device__ __host__ char operator()(T f0, T f1)
        {
            return f0 < f1;
        }
    };

    namespace base
    {
        template <
            typename T,
            typename BinaryOperation
        >
        void SortPerGroup(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& start, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& end, 
        uint elementsPerBlock, uint startStage, uint startStep, uint length, BinaryOperation op)
        {
            nutty::cuda::SortPerGroup(start, end, elementsPerBlock, startStage, startStep, length, op);
        }

        template<
            typename T, 
            typename BinaryOperation
        >
        void SortStep(Iterator<T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& start, uint grid, uint block, uint stage, uint step, BinaryOperation op, uint offset = 0)
        {
            nutty::cuda::SortStep(start, grid, block, stage, step, op, offset);
        }

        template <
            typename T,
            typename C,
            typename BinaryOperation
        >
        void Sort(Iterator<T, C>& start, Iterator<T, C>& end, BinaryOperation op)
        {
            uint length = (uint)Distance(start, end);

            uint elemCount = length;

            const uint maxElemsBlock = 512;

            uint maxElemsPerBlock = maxElemsBlock;

            if(elemCount >= maxElemsPerBlock)
            {
                elemCount = maxElemsPerBlock;
            }
            else
            {
                maxElemsPerBlock = elemCount;
            }

            SortPerGroup(start, end, elemCount, 2, 1, maxElemsPerBlock, op);

            elemCount = length;
            maxElemsPerBlock = maxElemsBlock;

            if(elemCount <= maxElemsPerBlock)
            {
                return;
            }

            uint blockSize = maxElemsBlock / 2;

            dim3 block;
            dim3 grid;

            block.x = blockSize;
            grid.x = (length / 2) / block.x;

            for(uint pow2stage = maxElemsPerBlock << 1; pow2stage <= length; pow2stage <<= 1)
            {
                for(uint step = pow2stage >> 1; step > 0; step = step >> 1)
                {
                    if((step << 1) <= maxElemsPerBlock)
                    {
                        SortPerGroup(start, end, maxElemsPerBlock, pow2stage, step, pow2stage, op);
                        break;
                    }
                    else
                    {
                        SortStep(start, block.x, grid.x, pow2stage, step, op);
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
        void SortKeyPerGroup(
        Iterator<
                K, nutty::base::Base_Buffer<K, nutty::DeviceContent<K>, nutty::CudaAllocator<K>>
                >& keyBegin, 
        Iterator<
                K, nutty::base::Base_Buffer<K, nutty::DeviceContent<K>, nutty::CudaAllocator<K>>
                >& keyEnd,
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& values, 
        uint elementsPerBlock, uint startStage, uint startStep, uint length, BinaryOperation op)
        {
            nutty::cuda::SortKeyPerGroup(keyBegin, keyEnd, values, elementsPerBlock, startStage, startStep, length, op);
        }

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
            typename T,
            typename C,
            typename K,
            typename KC,
            typename BinaryOperation
        >
        void Sort(Iterator<K, KC>& keyStart, Iterator<K, KC>& keyEnd, Iterator<T, C>& valuesBegin, BinaryOperation op)
        {
            uint length = (uint)Distance(keyStart, keyEnd);

            uint elemCount = length;

            const uint maxElemsBlock = 512;

            uint maxElemsPerBlock = maxElemsBlock;

            if(elemCount >= maxElemsPerBlock)
            {
                elemCount = maxElemsPerBlock;
            }
            else
            {
                maxElemsPerBlock = elemCount;
            }

            SortKeyPerGroup(keyStart, keyEnd, valuesBegin, elemCount, 2, 1, maxElemsPerBlock, op);

            elemCount = length;
            maxElemsPerBlock = maxElemsBlock;

            if(elemCount <= maxElemsPerBlock)
            {
                return;
            }

            uint blockSize = maxElemsBlock / 2;

            dim3 block;
            dim3 grid;

            block.x = blockSize;
            grid.x = (length / 2) / block.x;

            for(uint pow2stage = maxElemsPerBlock << 1; pow2stage <= length; pow2stage <<= 1)
            {
                for(uint step = pow2stage >> 1; step > 0; step = step >> 1)
                {
                    if((step << 1) <= maxElemsPerBlock)
                    {
                        SortKeyPerGroup(keyStart, keyEnd, valuesBegin, maxElemsPerBlock, pow2stage, step, pow2stage, op);
                        break;
                    }
                    else
                    {
                        SortKeyStep(keyStart, valuesBegin, block.x, grid.x, pow2stage, step, op);
                    }
                }
            }
        }
    }
}