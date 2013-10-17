#pragma once
#include "cuda/sort.h"
#include "base/Iterator.h"

namespace nutty
{
    template <
        typename T
    >
    struct BinaryDescending
    {
        __device__ char operator()(T f0, T f1)
        {
            return f0 > f1;
        }
    };

    template <
        typename T
    >
    struct BinaryAscending
    {
        __device__ char operator()(T f0, T f1)
        {
            return f0 < f1;
        }
    };

    template <
        typename T,
        typename C,
        typename BinaryOperation
    >
    void SortPerGroup(Iterator<T, C>& start, Iterator<T, C>& end, uint elementsPerBlock, uint startStage, uint startStep, uint length, BinaryOperation op)
    {
        nutty::cuda::SortPerGroup(start, end, elementsPerBlock, startStage, startStep, length, op);
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

        const uint maxElemsBloc = 512;

        uint maxElemsPerBlock = maxElemsBloc;

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
        maxElemsPerBlock = maxElemsBloc;

        if(elemCount <= maxElemsPerBlock)
        {
            return;
        }

        uint blockSize = maxElemsBloc / 2;

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
/*
                    cudah::ForEach(start, end, printOut);
                    cudaDeviceSynchronize();
                    printf("\n\n");*/
                    break;
                }
                else
                {
                    nutty::cuda::bitonicMergeSortStep(start, block.x, grid.x, pow2stage, step, op);
/*
                    cudaDeviceSynchronize();
                    cudah::ForEach(start, end, printOut);
                    printf("\n\n");*/
                }
            }
        }
    }

    template <
        typename T,
        typename BinaryOperation
    >
    void Sort(DeviceBuffer<T>& data, BinaryOperation op)
    {
        Sort(data.Begin(), data.End(), op);
    }

    template <
        typename T
    >
    void SortDescending(DeviceBuffer<T>& data)
    {
        BinaryDescending<T> op;
        Sort(data.Begin(), data.End(), op);
    }

    template <
        typename T
    >
    void SortAscending(DeviceBuffer<T>& data)
    {
        BinaryAscending<T> op;
        Sort(data.Begin(), data.End(), op);
    }
}