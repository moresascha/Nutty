
#pragma warning(disable: 4244)

#include "../Source/chimera/stdafx.h"
#include "Cudahu.h"

namespace cudahu
{
    cudah::cudah* m_pCudah = NULL;
    cudahu::CudahDevProp* m_pProperties = NULL;

    cudah::cudah* Init(LPCSTR projectDir, ID3D11Device* device /* = NULL */)
    {
        if(m_pCudah)
        {
            return m_pCudah;
        }
        cudah::Init(device);
        std::string dir = projectDir;
        dir += "Cudahu.ptx";
        m_pCudah = new cudah::cudah(dir.c_str());

        m_pProperties = new cudahu::CudahDevProp();
        cudahu::GetDeviceProp(m_pProperties, m_pCudah->GetDevice());

        return m_pCudah;
    }

    VOID Destroy(VOID)
    {
        SAFE_DELETE(m_pProperties);
        SAFE_DELETE(m_pCudah);
        cudah::Destroy();
    }

    VOID GetDeviceProp(CudahDevProp* prop, CUdevice device) 
    {
        cuDeviceGetProperties(prop, device);
        CUDA_DRIVER_SAFE_CALLING_SYNC(cuDeviceGetAttribute(&prop->concurrentKernels, CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS, device));
        CUDA_DRIVER_SAFE_CALLING_SYNC(cuDeviceGetAttribute(&prop->multiprocessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
        CUDA_DRIVER_SAFE_CALLING_SYNC(cuDeviceGetAttribute(&prop->L2Size, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device));
        CUDA_DRIVER_SAFE_CALLING_SYNC(cuDeviceGetAttribute(&prop->threadsPerMP, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device));

        prop->maxblocksPerMS = 8; //fermi todo
    }

    UINT GetThreadCount(UINT elements, UINT blockSize)
    {
        if(elements % blockSize == 0)
        {
            return elements;
        }
        return (elements / blockSize + 1) * blockSize;
    }

    VOID PrintArrayF(FLOAT* t, UINT l, UINT stride /* = -1 */)
    {
        for(UINT i = 0; i < l; ++i)
        {
            if(i != 0 && (i % stride == 0))
            {
                DEBUG_OUT("\n");
            }
            DEBUG_OUT_A("%f ", t[i]);
        }
        DEBUG_OUT("\n\n");
    }

    VOID reduceGPU(
        UINT count,
        UINT blockDim, 
        cudah::cuda_kernel kernel, 
        cudah::cuBuffer cdata, 
        cudah::cuBuffer cresult,
        UINT stride, 
        UINT memoryOffset)
    {
        kernel->m_blockDim = blockDim;
        kernel->m_gridDim = count / (2 * blockDim);
        kernel->m_shrdMemBytes = sizeof(float3) * blockDim * 2;
        VOID* args0[] = {&cdata->ptr, &cresult->ptr, &stride, &memoryOffset};
        kernel->m_ppArgs = args0;
        m_pCudah->CallKernel(kernel);
    }

    VOID reduce43GPU1S(
        cudah::cuBuffer data, 
        UINT count,
        UINT elementsPerBlock, 
        cudah::cuda_kernel kernel1,
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        reduceGPU(count, elementsPerBlock / 2, kernel1, data, result, stride, memoryOffset);

        if(cpu_result)
        {
            m_pCudah->ReadBuffer(result, cpu_result);
        }
    }

    VOID reduce43IndexedGPU(
        cudah::cuBuffer data, 
        cudah::cuBuffer index, 
        cudah::cuda_kernel kernel,
        UINT gridDim, 
        UINT blockDim,
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        kernel->m_blockDim = blockDim;
        kernel->m_gridDim = gridDim;
        kernel->m_shrdMemBytes = sizeof(float3) * kernel->m_blockDim.x * 2;
        VOID* args0[] = {&data->ptr, &result->ptr, &index->ptr, &stride, &memoryOffset};
        kernel->m_ppArgs = args0;
        m_pCudah->CallKernel(kernel);

        if(cpu_result)
        {
            m_pCudah->ReadBuffer(result, cpu_result);
        }
    }

    VOID reduce43MinIndexedGPU(
        cudah::cuBuffer data, 
        cudah::cuBuffer index, 
        UINT gridDim, 
        UINT blockDim,
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        cudah::cuda_kernel kernel = m_pCudah->GetKernel("_reduce_fromIndexMin4to3");
        reduce43IndexedGPU(data, index, kernel, gridDim, blockDim, result, stride, memoryOffset, cpu_result);
    }

    VOID reduce43MaxIndexedGPU(
        cudah::cuBuffer data, 
        cudah::cuBuffer index, 
        UINT gridDim, 
        UINT blockDim,
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        cudah::cuda_kernel kernel = m_pCudah->GetKernel("_reduce_fromIndexMax4to3");
        reduce43IndexedGPU(data, index, kernel, gridDim, blockDim, result, stride, memoryOffset, cpu_result);
    }

    VOID reduce43GPU(
        cudah::cuBuffer data, 
        UINT count, 
        UINT elementsPerBlock, 
        cudah::cuda_kernel kernel1, 
        cudah::cuda_kernel kenrel0, 
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        reduceGPU(count, elementsPerBlock / 2, kernel1, data, result, stride, memoryOffset);

        UINT elementsLeft = count / elementsPerBlock;

        if(elementsLeft > 1)
        {
            reduceGPU(elementsLeft, elementsLeft / 2, kenrel0, result, result, elementsLeft / 2, 0);
        }

        if(cpu_result)
        {
            m_pCudah->ReadBuffer(result, cpu_result);
        }
    }

    VOID reduceMin43GPU(
        cudah::cuBuffer data, 
        UINT count, 
        UINT elementsPerBlock, 
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        cudah::cuda_kernel pkmin = m_pCudah->GetKernel("_reduce_min4to3");
        reduce43GPU(data, count, elementsPerBlock, pkmin, m_pCudah->GetKernel("_reduce_min3to3"), result, stride, memoryOffset, cpu_result);
    }

    VOID reduceMax43GPU(
        cudah::cuBuffer 
        data,
        UINT count, 
        UINT elementsPerBlock, 
        cudah::cuBuffer result, 
        UINT stride,
        UINT memoryOffset, 
        float3* cpu_result)
    {
        cudah::cuda_kernel pkmax = m_pCudah->GetKernel("_reduce_max4to3");
        reduce43GPU(data, count, elementsPerBlock, pkmax, m_pCudah->GetKernel("_reduce_max3to3"), result, stride, memoryOffset, cpu_result);
    }

    VOID reduceMin43GPU1S(
        cudah::cuBuffer data, 
        UINT count, 
        UINT elementsPerBlock, 
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset, 
        float3* cpu_result)
    {
        cudah::cuda_kernel pkmin = m_pCudah->GetKernel("_reduce_min4to3");
        reduce43GPU1S(data, count, elementsPerBlock, pkmin, result, stride, memoryOffset, cpu_result);
    }

    VOID reduceMax43GPU1S(
        cudah::cuBuffer data,
        UINT count, 
        UINT elementsPerBlock, 
        cudah::cuBuffer result, 
        UINT stride, 
        UINT memoryOffset,
        float3* cpu_result)
    {
        cudah::cuda_kernel pkmax = m_pCudah->GetKernel("_reduce_max4to3");
        reduce43GPU1S(data, count, elementsPerBlock, pkmax, result, stride, memoryOffset, cpu_result);
    }

    VOID bitonicSortPerGroup(cudah::cuBuffer data, UINT elementsPerBlock, cudah::cuda_kernel kernel, UINT startStage, UINT startStep, UINT length)
    {
        kernel->m_blockDim.x = elementsPerBlock / 2;

        assert(m_pProperties->maxThreadsPerBlock >= kernel->m_blockDim.x);

        kernel->m_gridDim.x = (data->GetElementCount() / 2) / kernel->m_blockDim.x;
        kernel->m_shrdMemBytes = elementsPerBlock * data->GetElementSize();

        assert(m_pProperties->sharedMemPerBlock >= kernel->m_shrdMemBytes);

        kernel->SetKernelArg(0, data);
        kernel->SetKernelArg(1, startStage);
        kernel->SetKernelArg(2, startStep);
        kernel->SetKernelArg(3, length);

        m_pCudah->CallKernel(kernel);
    }

    VOID bitonicSort(cudah::cuBuffer data, cudah::cuda_kernel flatKernel, cudah::cuda_kernel shrdMemKernel, UINT blockSize, UINT offset, UINT length)
    {
        UINT elemCount = data->GetElementCount();
        UINT maxElemsPerBlock = m_pProperties->maxThreadsPerBlock/4;

        if(elemCount >= maxElemsPerBlock)
        {
            elemCount = maxElemsPerBlock;
        }

        bitonicSortPerGroup(data, elemCount, shrdMemKernel, 2, 1, maxElemsPerBlock);

        elemCount = data->GetElementCount();

        if(elemCount <= maxElemsPerBlock)
        {
            return;
        }

        flatKernel->m_blockDim.x = blockSize;
        flatKernel->m_gridDim.x = (length / 2) / flatKernel->m_blockDim.x;
        flatKernel->SetKernelArg(0, data);
        flatKernel->SetKernelArg(1, offset);

        for(UINT pow2stage = maxElemsPerBlock << 1; pow2stage <= length; pow2stage <<= 1)
        {
            flatKernel->SetKernelArg(2, pow2stage);
            for(UINT step = pow2stage >> 1; step > 0; step = step >> 1)
            {
                if((step << 1) <= maxElemsPerBlock)
                {
                    bitonicSortPerGroup(data, maxElemsPerBlock, shrdMemKernel, pow2stage, step, pow2stage);
                    break;
                }
                else
                {
                    flatKernel->SetKernelArg(3, step);
                    m_pCudah->CallKernel(flatKernel);
                }
            }
        }
    }

    VOID bitonicSortFlat(cudah::cuBuffer data, cudah::cuda_kernel flatKernel, UINT blockSize, UINT offset, UINT length)
    {
        flatKernel->m_blockDim.x = blockSize;

        flatKernel->m_gridDim.x = (length / 2) / flatKernel->m_blockDim.x;

        flatKernel->SetKernelArg(0, data);
        flatKernel->SetKernelArg(1, offset);
        for(UINT pow2stage = 2; pow2stage <= length; pow2stage <<= 1)
        {
            flatKernel->SetKernelArg(2, pow2stage);
            for(UINT step = pow2stage >> 1; step > 0; step = step >> 1)
            {
                flatKernel->SetKernelArg(3, step);
                m_pCudah->CallKernel(flatKernel);
            }
        }
    }

    VOID bitonicSortFloatPerGroup(cudah::cuBuffer data, UINT startStage, UINT elementsPerBlock, VOID* cpu_result)
    {
        cudah::cuda_kernel mergeSort = m_pCudah->GetKernel("bitonicMergeSortPerGroupFloat");
        bitonicSortPerGroup(data, elementsPerBlock, mergeSort, startStage, 1, elementsPerBlock);
        if(cpu_result)
        {
            m_pCudah->ReadBuffer(data, cpu_result);
        }
    }

    VOID bitonicSortFloatPerGroup(cudah::cuBuffer data, VOID* cpu_result)
    {
        bitonicSortFloatPerGroup(data, 1, data->GetElementCount(), cpu_result);
    }

    VOID bitonicSortFloat(cudah::cuBuffer data, VOID* cpu_result)
    {
        cudah::cuda_kernel mergeSortFlat = m_pCudah->GetKernel("bitonicMergeSortStepFloat");
        cudah::cuda_kernel mergeSortShared = m_pCudah->GetKernel("bitonicMergeSortPerGroupFloat");
        bitonicSort(data, mergeSortFlat, mergeSortShared, 256, 0, data->GetElementCount());
        if(cpu_result)
        {
            m_pCudah->ReadBuffer(data, cpu_result);
        }
    }

    VOID bitonicSortFloatFlat(cudah::cuBuffer data, VOID* cpu_result)
    {
        cudah::cuda_kernel mergeSortFlat = m_pCudah->GetKernel("bitonicMergeSortStepFloat");
        bitonicSortFlat(data, mergeSortFlat, 256, 0, data->GetElementCount());
        if(cpu_result)
        {
            m_pCudah->ReadBuffer(data, cpu_result);
        }
    }
}