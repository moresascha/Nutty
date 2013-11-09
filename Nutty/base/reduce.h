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
            typename T,
            typename BinaryOperation
        >
        void Reduce(
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& begin, 
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& end,
        BinaryOperation op)
        {
            size_t d = Distance(begin, end);

            uint elementsPerBlock = 512;
            
            assert(d > 1);

            //todo: reduce small sets on the cpu

            nutty::cuda::Reduce(begin(), begin(), d, op, elementsPerBlock);

            if(d < elementsPerBlock)
            {
                return;
            }

            uint rest = (d % elementsPerBlock);

            uint grid = d / elementsPerBlock;

            if(rest > 0)
            {
                nutty::cuda::Reduce(begin() + grid, begin(), rest, op, rest, elementsPerBlock * grid);
            } 

            UINT elementsLeft = (uint)d / elementsPerBlock + (rest ? 1 : 0);

            if(elementsLeft > 1)
            {
                Reduce(begin, begin + elementsLeft, op);
            }
        }

        template <
            typename T,
            typename T_ID,
            typename BinaryOperation
        >
        void ReduceIndexed(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dstBegin,
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcEnd,
        Iterator<
                T_ID, nutty::base::Base_Buffer<T_ID, nutty::DeviceContent<T_ID>, nutty::CudaAllocator<T_ID>>
                >& indexBegin,
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

        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dstBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::HostContent<T>, nutty::DefaultAllocator<T>>
                >& dstEnd,

        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcEnd,

        BinaryOperation op)
        {
            size_t dd = Distance(srcBegin, srcEnd);

            DeviceBuffer<T> deviceDest(dd);
            
            assert(0);

            //nutty::cuda::Reduce(deviceDest.Begin()(), srcBegin(), dd, op);

            size_t dh = Distance(dstBegin, dstEnd);

            nutty::Copy(dstBegin, deviceDest.Begin(), dh);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void Reduce(
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dstBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& dstEnd,

        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcBegin, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& srcEnd,

        BinaryOperation op)
        {
            size_t dd = Distance(srcBegin, srcEnd);
            assert(0);
            //DeviceBuffer<T> deviceDest(dd);

            //nutty::cuda::Reduce(dstBegin(), srcBegin(), dd, op);

            //size_t dh = Distance(dstBegin, dstEnd);

            //nutty::Copy(dstBegin, deviceDest.Begin(), dh);
        }

        template <
            typename T,
            typename BinaryOperation
        >
        T Reduce(
        BinaryOperation op,
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& begin, 
        Iterator<
            T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
            >& end
        )
        {
            HostBuffer<T> host(1);
            assert(0);
            //nutty::base::Reduce(host.Begin(), host.End(), begin, end, op);
            return host[0];
        }
    }
}