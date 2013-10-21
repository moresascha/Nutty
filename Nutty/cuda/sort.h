#pragma once
#include "Globals.cuh"

namespace nutty
{
    namespace cuda
    {
        //key value sort

        template <
            template <typename, typename> class KVPair, class K, class V,
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortKeyStep(KVPair<K, V>* kv, uint stage, uint step, uint id, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (id / step);
            uint second = first + step;

            uint bankOffset = (id % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            KVPair<K, V> kv0 = kv[offset + first];
            KVPair<K, V> kv1 = kv[offset + second];

            char dir = (((2 * GlobalId ) / stage) & 1);  //order & ((((2 * id ) / stage) & 1) + 1);
            char cmp = _cmp_func(kv0.v, kv1.v);

            if((!dir & cmp) | (dir & !cmp))
            {
                kv[offset + first].k = kv1.k;
                kv[offset + second].k = kv0.k;
            }
        }

        template <
            typename V,
            typename K,
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortKeyStep(V* v, K* k, uint stage, uint step, uint id, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (id / step);
            uint second = first + step;

            uint bankOffset = (id % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            K k0 = k[offset + first];
            K k1 = k[offset + second];

            V v0 = v[k0];
            V v1 = v[k1];

            char dir = (((2 * GlobalId ) / stage) & 1);  //order & ((((2 * id ) / stage) & 1) + 1);
            char cmp = _cmp_func(v0, v1);

            if((!dir & cmp) | (dir & !cmp))
            {
                k[offset + first] = k1;
                k[offset + second] = k0;
            }
        }

        template <
            typename T, 
            typename K,
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortStep(T* v, K* k, uint stage, uint step, BinaryOperation _cmp_func, uint offset = 0)
        {
            __bitonicMergeSortKeyStep(v, k, stage, step, GlobalId, _cmp_func, offset);
        }

        template <
            typename T,
            typename K,
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortKeyPerGroup(T* v, K* key, uint startStage, uint startStep, uint length, BinaryOperation _cmp_func)
        {
            //char order = blockIdx.x % 2 + 1;
            uint tId = threadIdx.x;
            uint elementsPerBlock = 2 * blockDim.x;
            uint i = blockIdx.x * elementsPerBlock + 2 * tId;

            ShrdMemory<KVPair<K, T>> shrdMem;
            KVPair<K, T>* shrd = shrdMem.Ptr();

            K k0 = key[i + 0];
            K k1 = key[i + 1];
            T v0 = v[k0];
            T v1 = v[k1];

            shrd[2 * tId + 0] = KVPair<K, T>(k0, v0);
            shrd[2 * tId + 1] = KVPair<K, T>(k1, v1);

            uint step = startStep;

            for(uint stage = startStage; stage <= length;)
            {
                for(;step > 0; step = step >> 1)
                {
                    __syncthreads();
                    __bitonicMergeSortKeyStep(shrd, stage, step, tId, _cmp_func);
                }
                stage <<= 1;
                step = stage >> 1;
            }

            key[i + 0] = shrd[2 * tId + 0].k;
            key[i + 1] = shrd[2 * tId + 1].k;
        }

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
            uint d = (uint)Distance(keyBegin, keyEnd);
            dim3 block = elementsPerBlock / 2; 
            dim3 grid = (d / 2) / block.x;

            uint shrdMem = elementsPerBlock * sizeof(KVPair<K,T>);

            bitonicMergeSortKeyPerGroup
                <<<grid, block, shrdMem>>>
                (
                values(), keyBegin(), startStage, startStep, length, op
                );
        }

        template <
            typename T,
            typename K,
            typename BinaryOperation
        >
        void SortKeyStep(
        Iterator<
                K, nutty::base::Base_Buffer<K, nutty::DeviceContent<K>, nutty::CudaAllocator<K>>
                >& keys, 
        Iterator<
                T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
                >& values, 
        uint grid, uint block, uint stage, uint step, BinaryOperation op, uint offset = 0)
        {
            bitonicMergeSortStep<<<grid, block>>>
            (
            values(), keys(), stage, step, op
            );
        }

        //value sort

        template <
            typename T, 
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortStep(T* v, uint stage, uint step, uint id, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (id / step);
            uint second = first + step;

            uint bankOffset = (id % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            T n0 = v[offset + first];

            T n1 = v[offset + second];

            char dir = (((2 * GlobalId ) / stage) & 1);  //order & ((((2 * id ) / stage) & 1) + 1);
            char cmp = _cmp_func(n0, n1);

            if((!dir & cmp) | (dir & !cmp))
            {
                v[offset + first] = n1;
                v[offset + second] = n0;
            }
        }

        template <
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortStep(T* v, uint stage, uint step, BinaryOperation _cmp_func, uint offset = 0)
        {
            __bitonicMergeSortStep(v, stage, step, GlobalId, _cmp_func, offset);
        }

        template <
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortPerGroup(T* g_values, uint startStage, uint startStep, uint length, BinaryOperation _cmp_func)
        {
            //char order = blockIdx.x % 2 + 1;
            uint tId = threadIdx.x;
            uint elementsPerBlock = 2 * blockDim.x;
            uint i = blockIdx.x * elementsPerBlock + 2 * tId;

            ShrdMemory<T> shrdMem;
            T* shrd = shrdMem.Ptr();

            shrd[2 * tId + 0] = g_values[i + 0];
            shrd[2 * tId + 1] = g_values[i + 1];

            uint step = startStep;

            for(uint stage = startStage; stage <= length;)
            {
                for(;step > 0; step = step >> 1)
                {
                    __syncthreads();
                    __bitonicMergeSortStep(shrd, stage, step, tId, _cmp_func);
                }
                stage <<= 1;
                step = stage >> 1;
            }

            g_values[i + 0] = shrd[2 * tId + 0];
            g_values[i + 1] = shrd[2 * tId + 1];
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void SortPerGroup(
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& valuesBegin, 
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>> 
        >& valuesEnd, 
        uint elementsPerBlock, uint startStage, uint startStep, uint length, BinaryOperation op)
        {
            uint d = (uint)Distance(valuesBegin, valuesEnd);
            dim3 block = elementsPerBlock / 2; 
            dim3 grid = (d / 2) / block.x;

            uint shrdMem = elementsPerBlock * sizeof(T);

            bitonicMergeSortPerGroup
                <<<grid, block, shrdMem>>>
                (
                valuesBegin(),startStage, startStep, length, op
                );
        }

        template <
            typename T,
            typename BinaryOperation
        >
        void SortStep(
        Iterator<
        T, nutty::base::Base_Buffer<T, nutty::DeviceContent<T>, nutty::CudaAllocator<T>>
        >& values, 
        uint grid, uint block, uint stage, uint step, BinaryOperation op, uint offset = 0)
        {
            __bitonicMergeSortStep
                <<<grid, block>>>
                (
                values(), stage, step, op
                );
        }
    }
}