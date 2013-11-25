#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"

namespace nutty
{
    namespace cuda
    {
        /************************************************************************/
        /* VALUE SORT                                                           */
        /************************************************************************/
        template <
            typename T, 
            typename BinaryOperation
        >
        __device__ char __cmp(T t0, T t1, BinaryOperation _op, uint stage)
        {
            char dir = (((2 * GlobalId ) / stage) & 1);
            char cmp = _op(t0, t1);
            return (!dir & cmp) | (dir & !cmp);
        }

        template <
            typename T, 
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortStep(T* v, uint stage, uint step, uint tid, uint length, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (tid / step);
            uint second = first + step;
            
            uint bankOffset = (tid % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            uint gsecond = (step << 1) * (GlobalId / step) + step;
            gsecond += bankOffset * isBankOffset;
            
            uint start = (step<<1) * (GlobalId/(step));

            if(start + (step<<1) > length)
            {
                char dir = (((2 * GlobalId ) / stage) & 1);
                if(!dir && stage == (step<<1))
                {
                    first += (step<<1) - (length%(step<<1));
                }
            }

            if(offset + gsecond < length)
            {
                T n0 = v[offset + first];

                T n1 = v[offset + second];

                if(__cmp(n0, n1, _cmp_func, stage))
                {
                    v[offset + second] = n0;
                    v[offset + first] = n1;
                }
            }
        }

        template <
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortStep(T* v, uint stage, uint step, uint length, BinaryOperation _cmp_func, uint offset = 0)
        {
            __bitonicMergeSortStep(v, stage, step, GlobalId, length, _cmp_func, offset);
        }

        template <
            typename T, 
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortPerGroup(T* g_values, uint startStage, uint endStage, uint startStep, uint length, BinaryOperation _cmp_func)
        {
            uint tId = threadIdx.x;
            uint i = 2 * GlobalId;

            if(i >= length)
            {
                return;
            }

            ShrdMemory<T> shrdMem;
            T* shrd = shrdMem.Ptr();

            shrd[2 * tId + 0] = g_values[i + 0];

            if(i+1 < length)
            {
                shrd[2 * tId + 1] = g_values[i + 1];
            }
            
            uint step = startStep;
          
            for(uint stage = startStage; stage <= endStage;)
            {
                for(;step > 0; step = step >> 1)
                {
                    __syncthreads();
                    __bitonicMergeSortStep(shrd, stage, step, tId, length, _cmp_func);
                }

                stage <<= 1;
                step = stage >> 1;
            }

            g_values[i + 0] = shrd[2 * tId + 0];

            if(i+1 < length)
            {
                g_values[i + 1] = shrd[2 * tId + 1];
            }
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
        uint elementsPerBlock, uint startStage, uint endStage, uint startStep, uint length, BinaryOperation op)
        {
            dim3 block = (elementsPerBlock + elementsPerBlock%2)/2;//1 << nutty::getmsb((elementsPerBlock + (elementsPerBlock%2))/2);
            dim3 grid = getCudaGrid(length, elementsPerBlock);

            uint shrdMem = elementsPerBlock * sizeof(T);

            bitonicMergeSortPerGroup
                <<<grid, block, shrdMem>>>
                (
                valuesBegin(), startStage, endStage, startStep, length, op
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
        uint grid, uint block, uint stage, uint step, uint length, BinaryOperation op, uint offset = 0)
        {
            bitonicMergeSortStep
                <<<grid, block>>>
                (
                values(), stage, step, length, op, offset
                );
        }

        /************************************************************************/
        /* KEY VALUE SORT                                                       */
        /************************************************************************/
                /*
        template <
        template <typename, typename> class KVPair, class K, class V,
        typename BinaryOperation
        >
        __device__ void __bitonicMergeSortKeyStep(KVPair<K, V>* kv, uint stage, uint step, uint id, BinaryOperation _cmp_func, uint offset = 0)
        {

        }
        */

        template <
            typename V,
            typename K,
            typename BinaryOperation
        >
        __device__ void __bitonicMergeSortKeyStep(V* v, K* k, uint stage, uint step, uint id, uint length, BinaryOperation _cmp_func, uint offset = 0)
        {
            uint first = (step << 1) * (id / step);
            uint second = first + step;

            uint bankOffset = (id % step);
            char isBankOffset = bankOffset > 0;

            first += bankOffset * isBankOffset;
            second += bankOffset * isBankOffset;

            uint gsecond = (step << 1) * (GlobalId / step) + step;
            gsecond += bankOffset * isBankOffset;

            uint start = (step<<1) * (GlobalId/(step));

            if(start + (step<<1) > length)
            {
                char dir = (((2 * GlobalId ) / stage) & 1);
                if(!dir && stage == (step<<1))
                {
                    first += (step<<1) - (length%(step<<1));
                }
            }

            if(offset + gsecond < length)
            {
                K k0 = k[offset + first];

                K k1 = k[offset + second];

                V v0 = v[k0];

                V v1 = v[k1];

                if(__cmp(v0, v1, _cmp_func, stage))
                {
                    k[offset + first] = k1;
                    k[offset + second] = k0;
                    /*v[offset + first] = v1;
                    v[offset + second] = v0;*/
                }
            }
        }

        template <
            typename T, 
            typename K,
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortKeyStep(T* v, K* k, uint stage, uint step, BinaryOperation _cmp_func, uint offset = 0)
        {
            __bitonicMergeSortKeyStep(v, k, stage, step, GlobalId, _cmp_func, offset);
        }

        template <
            typename T,
            typename K,
            typename BinaryOperation
        >
        __global__ void bitonicMergeSortKeyPerGroup(T* v, K* k, uint startStage, uint endStage, uint startStep, uint length, BinaryOperation _cmp_func)
        {
            uint tId = threadIdx.x;
            uint i = 2 * GlobalId;

            if(i >= length)
            {
                return;
            }

            /*ShrdMemory<KVPair<K, T>> shrdMem;
            KVPair<K, T>* shrd = shrdMem.Ptr();

            K k0 = key[i + 0];
            T v0 = v[k0];

            shrd[2 * tId + 0] = KVPair<K, T>(k0, v0);
            if(i+1 < length)
            {
                K k1 = key[i + 1];
                T v1 = v[k1];
                shrd[2 * tId + 1] = KVPair<K, T>(k1, v1);
            }*/

            uint step = startStep;

            for(uint stage = startStage; stage <= endStage;)
            {
                for(;step > 0; step = step >> 1)
                {
                    __syncthreads();
                    __bitonicMergeSortKeyStep(v, k, stage, step, tId, length, _cmp_func);
                }
                stage <<= 1;
                step = stage >> 1;
            }

            /*key[i + 0] = shrd[2 * tId + 0].k;
            v[i + 0] = shrd[2 * tId + 0].v;
            if(i+1 < length)
            {
                key[i + 1] = shrd[2 * tId + 1].k;
                v[i + 1] = shrd[2 * tId + 1].v;
            }*/
        }

        template <
            typename IteratorKey,
            typename IteratorData,
            typename BinaryOperation
        >
        void SortKeyPerGroup(
        IteratorKey& keyBegin, 
        IteratorKey& keyEnd, 
        IteratorData& values, 
        uint elementsPerBlock, uint startStage, uint endstage, uint startStep, uint length, BinaryOperation op)
        {
            dim3 block = (elementsPerBlock + elementsPerBlock%2)/2;
            dim3 grid = getCudaGrid(length, elementsPerBlock);

            uint shrdMem = 0;//elementsPerBlock * sizeof(KVPair<K,T>);

            bitonicMergeSortKeyPerGroup
                <<<grid, block, shrdMem>>>
                (
                values(), keyBegin(), startStage, endstage, startStep, length, op
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
            bitonicMergeSortKeyStep<<<grid, block>>>
                (
                values(), keys(), stage, step, op
                );
        }
    }
}