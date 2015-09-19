#pragma once
#include "Globals.cuh"
#include "cuda_helper.h"
#include <device_functions.h>

__device__ __forceinline__ uint __warp_scan(int val, volatile uint* sdata)
{
    uint idx = 2 * threadIdx.x - (threadIdx.x & (warpSize-1));
    sdata[idx] = 0;
    idx += warpSize;
    uint t = sdata[idx] = val;
    sdata[idx] = t = t + sdata[idx-1];
    sdata[idx] = t = t + sdata[idx-2];
    sdata[idx] = t = t + sdata[idx-4];
    sdata[idx] = t = t + sdata[idx-8];
    sdata[idx] = t = t + sdata[idx-16];

    return sdata[idx-1];
}

__device__ __forceinline__ unsigned int __lanemasklt()
{
    const unsigned int lane = threadIdx.x & (warpSize-1);
    return (1<<(lane)) - 1;
}

template <typename T>
__device__ unsigned int __warpprefixsums(T p)
{
    const unsigned int mask = __lanemasklt();
    unsigned int b = __ballot((bool)p);
    return __popc(b & mask);
}

template <typename T>
__device__ T __blockBinaryPrefixSums(T* sdata, int x) 
{ 
    int warpPrefix = __warpprefixsums(x);
    int idx = threadIdx.x;
    int warpIdx = idx / warpSize;
    int laneIdx = idx & (warpSize-1); 

    if(laneIdx == warpSize-1) 
    {
        sdata[warpIdx] = warpPrefix + x; 
    }

    __syncthreads(); 

    if(idx < warpSize)
    {
        sdata[idx] = __warp_scan(sdata[idx], sdata); 
    }

    __syncthreads();

    return sdata[warpIdx] + warpPrefix;
}

__device__ __forceinline__  int __laneid(void)
{
    return threadIdx.x & (warpSize-1);
}

template < 
    uint width
>
__device__ int __blockScan(uint* sums, int value)
{
    int warp_id = threadIdx.x / warpSize;
    int lane_id = __laneid();
#pragma unroll
    for (uint i=1; i<width; i*=2)
    {
        int n = __shfl_up(value, i, width);

        if (lane_id >= i) value += n;
    }

    // value now holds the scan value for the individual thread
    // next sum the largest values for each warp

    // write the sum of the warp to smem
    if (threadIdx.x % warpSize == warpSize-1)
    {
        sums[warp_id] = value;
    }

    __syncthreads();

    //
    // scan sum the warp sums
    // the same shfl scan operation, but performed on warp sums
    //
    if (warp_id == 0)
    {
        int warp_sum = sums[lane_id];

        for (int i=1; i<width; i*=2)
        {
            int n = __shfl_up(warp_sum, i, width);

            if (__laneid() >= i) warp_sum += n;
        }

        sums[lane_id] = warp_sum;
    }

    __syncthreads();

    int blockSum = 0;

    if (warp_id > 0)
    {
        blockSum = sums[warp_id-1];
    }

    value += blockSum;

    return value;
}

//LOCAL_SCAN_SIZE + LOCAL_SCAN_SIZE / NUM_BANKS
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) / NUM_BANKS)

template <
    typename T,
    uint blockSize,
    uint LOCAL_SCAN_SIZE
>
__global__ void __binaryGroupScan4(const T* __restrict g_data, uint4* scanned, uint* sums, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    uint id = threadIdx.x;

    int globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;
    int ai = id;
    int bi = id + (LOCAL_SCAN_SIZE / 2);

    T a = g_data[globalOffset + ai];
    T b = g_data[globalOffset + bi];
    
    uint partSumA1 = a.y + a.x;
    uint partSumA2 = a.z + partSumA1;

    uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w);

    uint leftSideSum = shrdMem[blockSize/32-1];

    uint partSumB1 = b.y +  b.x;
    uint partSumB2 = b.z + partSumB1;

    __syncthreads();

    uint sum1 = leftSideSum + __blockScan<blockSize>(shrdMem, partSumB2 + b.w) - (partSumB2 + b.w);


    scanned[globalOffset + ai].x = sum0;
    scanned[globalOffset + ai].y = sum0 + a.x;
    scanned[globalOffset + ai].z = sum0 + partSumA1;
    scanned[globalOffset + ai].w = sum0 + partSumA2;

    scanned[globalOffset + bi].x = sum1;
    scanned[globalOffset + bi].y = sum1 + b.x;
    scanned[globalOffset + bi].z = sum1 + partSumB1;
    scanned[globalOffset + bi].w = sum1 + partSumB2;

    if(threadIdx.x == blockDim.x-1 && sums)
    {
        sums[blockIdx.x] = sum1 + partSumB2 + b.w;
    }
}

__device__ void __forceinline__ __cpyN(void* dst, void* src, int bytes)
{
    memcpy(dst, src, bytes);
}

template <
    uint blockSize,
    uint LOCAL_SCAN_SIZE
>
__global__ void __binaryGroupScan4Test(const uchar4* __restrict g_data, uint4* scanned, uint* sums, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    int globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;

    if(globalOffset >= N)
    {
        return;
    }

    int ai = threadIdx.x;
    int bi = threadIdx.x + (LOCAL_SCAN_SIZE / 2);

    uchar4 a = {0,0,0,0};
    uchar4 b = {0,0,0,0};

    if(4 * (globalOffset + ai) + 3 < N)
    {
        a = g_data[globalOffset + ai];
    }    
    else
    {
        int elemsLeft = N - 4 * (globalOffset + ai); 
        if(elemsLeft > 0 && elemsLeft < 4)
        {
            memcpy((void*)&a, (void*)(g_data + globalOffset + ai), elemsLeft);
        }
    }

    if(4 * (globalOffset + bi) + 3 < N)
    {
        b = g_data[globalOffset + bi];
    }
    else
    {
        int elemsLeft = N - 4 * (globalOffset + bi);
        if(elemsLeft > 0 && elemsLeft < 4)
        {
            memcpy((void*)&b, (void*)(g_data + globalOffset + bi), elemsLeft);
        }
    }

    uint partSumA1 = a.y + a.x;
    uint partSumA2 = a.z + partSumA1;

    uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w);

    uint leftSideSum = shrdMem[blockSize/32-1];

    uint partSumB1 = b.y +  b.x;
    uint partSumB2 = b.z + partSumB1;

    __syncthreads();

    uint sum1 = leftSideSum + __blockScan<blockSize>(shrdMem, partSumB2 + b.w) - (partSumB2 + b.w);

    if(4 * (globalOffset + ai) + 3 < N)
    {
        scanned[globalOffset + ai].x = sum0;
        scanned[globalOffset + ai].y = sum0 + a.x;
        scanned[globalOffset + ai].z = sum0 + partSumA1;
        scanned[globalOffset + ai].w = sum0 + partSumA2;
    }
    else
    {
        int elemsLeft = N - 4 * (globalOffset + ai); 
        if(elemsLeft > 0 && elemsLeft < 4)
        {
            uint3 _a;
            _a.x = sum0;
            _a.y = sum0 + a.x;
            _a.z = sum0 + partSumA1;
            memcpy((void*)(scanned + globalOffset + ai), (void*)&_a, 4 *elemsLeft);
        }
    }

    if(4 * (globalOffset + bi) + 3 < N)
    {
        scanned[globalOffset + bi].x = sum1;
        scanned[globalOffset + bi].y = sum1 + b.x;
        scanned[globalOffset + bi].z = sum1 + partSumB1;
        scanned[globalOffset + bi].w = sum1 + partSumB2;
    }
    else
    {
        int elemsLeft = N - 4 * (globalOffset + bi); 
        if(elemsLeft > 0 && elemsLeft < 4)
        {
            uint3 _a;
            _a.x = sum1;
            _a.y = sum1 + b.x;
            _a.z = sum1 + partSumB1;
            memcpy((void*)(scanned + globalOffset + bi), (void*)&_a, 4 * elemsLeft);
        }
    }

    if(threadIdx.x == blockDim.x-1 && sums)
    {
        sums[blockIdx.x] = sum1 + partSumB2 + b.w;
    }
}

template <
    uint blockSize,
    uint LOCAL_SCAN_SIZE
>
__global__ void __groupScan4TestAll(const uint4* __restrict g_data, uint4* scanned, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    uint id = threadIdx.x;

    int globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;
    int ai = id;
    int bi = id + (LOCAL_SCAN_SIZE / 2);

    __shared__ uint s_gSum;
    
    if(!threadIdx.x) s_gSum = 0;

    for(uint offset = 0; offset < N; offset += LOCAL_SCAN_SIZE)
    {
        uint4 a = {0,0,0,0};
        uint4 b = {0,0,0,0};
        globalOffset += offset;
        if(4 * (globalOffset + ai) + 3 < N)
        {
            a = g_data[globalOffset + ai];
        }    
        else
        {
            int elemsLeft = N - 4 * (globalOffset + ai); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                memcpy((void*)&a, (void*)(g_data + globalOffset + ai), 4*elemsLeft);
            }
        }

        if(4 * (globalOffset + bi) + 3 < N)
        {
            b = g_data[globalOffset + bi];
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + bi); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                memcpy((void*)&b, (void*)(g_data + globalOffset + bi), 4*elemsLeft);
            }
        }

        uint partSumA1 = a.y + a.x;
        uint partSumA2 = a.z + partSumA1;

        uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w) + s_gSum;

        uint leftSideSum = shrdMem[blockSize/32-1];

        uint partSumB1 = b.y +  b.x;
        uint partSumB2 = b.z + partSumB1;

        __syncthreads();

        uint sum1 = leftSideSum + __blockScan<blockSize>(shrdMem, partSumB2 + b.w) - (partSumB2 + b.w) + s_gSum;

        if(4 * (globalOffset + ai) + 3 < N)
        {
            scanned[globalOffset + ai].x = sum0;
            scanned[globalOffset + ai].y = sum0 + a.x;
            scanned[globalOffset + ai].z = sum0 + partSumA1;
            scanned[globalOffset + ai].w = sum0 + partSumA2;
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + ai); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                uint3 _a;
                _a.x = sum0;
                _a.y = sum0 + a.x;
                _a.z = sum0 + partSumA1;
                memcpy((void*)(scanned + globalOffset + ai), (void*)&_a, 4*elemsLeft);
            }
        }

        if(4 * (globalOffset + bi) + 3 < N)
        {
            scanned[globalOffset + bi].x = sum1;
            scanned[globalOffset + bi].y = sum1 + b.x;
            scanned[globalOffset + bi].z = sum1 + partSumB1;
            scanned[globalOffset + bi].w = sum1 + partSumB2;
        }
        else
        {
            int elemsLeft = N - 4 * (globalOffset + bi); 
            if(elemsLeft > 0 && elemsLeft < 4)
            {
                uint3 _a;
                _a.x = sum1;
                _a.y = sum1 + b.x;
                _a.z = sum1 + partSumB1;
                memcpy((void*)(scanned + globalOffset + bi), (void*)&_a, 4*elemsLeft);
            }
        }

        if(threadIdx.x == blockDim.x-1)
        {
            s_gSum += sum1 + partSumB2 + b.w;
        }
    }
}

#define FIRST_BYTE(V)  ((V & (1 <<  0)) > 0)
#define SECOND_BYTE(V) ((V & (1 <<  8)) > 0)
#define THIRD_BYTE(V)  ((V & (1 << 16)) > 0)
#define FOURTH_BYTE(V) ((V & (1 << 24)) > 0)

#define SUM_BYTES(V) (FIRST_BYTE(V) + SECOND_BYTE(V) + THIRD_BYTE(V) + FOURTH_BYTE(V))

#define MAKE_SCANNED_INT(V)

struct uint16
{
    uint4 x;
    uint4 y;
    uint4 z;
    uint4 w;
};

template <
    uint blockSize,
    uint LOCAL_SCAN_SIZE
>
__global__ void __binaryGroupScan4B(const uint4* __restrict g_data, uint16* scanned, uint* sums, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    //__shared__ uint4 transposed[4*2*blockSize];

    uint id = threadIdx.x;

    int globalOffset = blockIdx.x * LOCAL_SCAN_SIZE;
    int ai = id;
    int bi = id + (LOCAL_SCAN_SIZE / 2);

    uint4 a = g_data[globalOffset + ai];
    uint4 b = g_data[globalOffset + bi];

    uint partSumA1 = SUM_BYTES(a.y) + SUM_BYTES(a.x);
    uint partSumA2 = SUM_BYTES(a.z) + partSumA1;

    uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + SUM_BYTES(a.w)) - (partSumA2 + SUM_BYTES(a.w));

    uint leftSideSum = shrdMem[blockSize/32-1];

    uint partSumB1 = SUM_BYTES(b.y) + SUM_BYTES(b.x);
    uint partSumB2 = SUM_BYTES(b.z) + partSumB1;

    __syncthreads();

    uint sum1 = leftSideSum + __blockScan<blockSize>(shrdMem, partSumB2 + SUM_BYTES(b.w)) - (partSumB2 + SUM_BYTES(b.w));

    uint16 ra, rb;
    
    ra.x.x = sum0;
    ra.x.y = sum0 + FIRST_BYTE(a.x);
    ra.x.z = sum0 + FIRST_BYTE(a.x) + SECOND_BYTE(a.x);
    ra.x.w = sum0 + FIRST_BYTE(a.x) + SECOND_BYTE(a.x) + THIRD_BYTE(a.x);

    ra.y.x = sum0 + SUM_BYTES(a.x);
    ra.y.y = sum0 + FIRST_BYTE(a.y) + SUM_BYTES(a.x);
    ra.y.z = sum0 + FIRST_BYTE(a.y) + SECOND_BYTE(a.y) + SUM_BYTES(a.x);
    ra.y.w = sum0 + FIRST_BYTE(a.y) + SECOND_BYTE(a.y) + THIRD_BYTE(a.y) + SUM_BYTES(a.x);

    ra.z.x = sum0 + partSumA1;
    ra.z.y = sum0 + FIRST_BYTE(a.z) + partSumA1;
    ra.z.z = sum0 + FIRST_BYTE(a.z) + SECOND_BYTE(a.z) + partSumA1;
    ra.z.w = sum0 + FIRST_BYTE(a.z) + SECOND_BYTE(a.z) + THIRD_BYTE(a.z) + partSumA1;

    ra.w.x = sum0 + partSumA2;
    ra.w.y = sum0 + FIRST_BYTE(a.w) + partSumA2;
    ra.w.z = sum0 + FIRST_BYTE(a.w) + SECOND_BYTE(a.w) + partSumA2;
    ra.w.w = sum0 + FIRST_BYTE(a.w) + SECOND_BYTE(a.w) + THIRD_BYTE(a.w) + partSumA2;

    rb.x.x = sum1;
    rb.x.y = sum1 + FIRST_BYTE(b.x);
    rb.x.z = sum1 + FIRST_BYTE(b.x) + SECOND_BYTE(b.x);
    rb.x.w = sum1 + FIRST_BYTE(b.x) + SECOND_BYTE(b.x) + THIRD_BYTE(b.x);

    rb.y.x = sum1 + SUM_BYTES(b.x);
    rb.y.y = sum1 + FIRST_BYTE(b.y) + SUM_BYTES(b.x);
    rb.y.z = sum1 + FIRST_BYTE(b.y) + SECOND_BYTE(b.y) + SUM_BYTES(b.x);
    rb.y.w = sum1 + FIRST_BYTE(b.y) + SECOND_BYTE(b.y) + THIRD_BYTE(b.y) + SUM_BYTES(b.x);

    rb.z.x = sum1 + partSumA1;
    rb.z.y = sum1 + FIRST_BYTE(b.z) + partSumB1;
    rb.z.z = sum1 + FIRST_BYTE(b.z) + SECOND_BYTE(b.z) + partSumB1;
    rb.z.w = sum1 + FIRST_BYTE(b.z) + SECOND_BYTE(b.z) + THIRD_BYTE(b.z) + partSumB1;

    rb.w.x = sum1 + partSumB2;
    rb.w.y = sum1 + FIRST_BYTE(b.w) + partSumB2;
    rb.w.z = sum1 + FIRST_BYTE(b.w) + SECOND_BYTE(b.w) + partSumB2;
    rb.w.w = sum1 + FIRST_BYTE(b.w) + SECOND_BYTE(b.w) + THIRD_BYTE(b.w) + partSumB2;

    scanned[globalOffset + ai] = ra;
    scanned[globalOffset + bi] = rb;

    if(threadIdx.x == blockDim.x-1 && sums)
    {
        sums[blockIdx.x] = sum1 + (partSumB2 + SUM_BYTES(b.w));
    }
}

template <
    uint blockSize
>
__global__ void __binaryGroupScanN(const uchar4* __restrict g_data, uint4* scanned, uint* sums, uint N, uint scanLineSize)
{
    __shared__ uint shrdMem[blockSize/32];

    uint globalOffset = blockIdx.x * scanLineSize;

    uint gsum = 0;

    uchar4 a = make_uchar4(0,0,0,0);

    uint ai = threadIdx.x;

//     if(globalOffset + threadIdx.x < N)
//     {
//         a = g_data[globalOffset + threadIdx.x];
//     }

    //uchar4 nextElem = make_uchar4(0,0,0,0);

#pragma unroll
    for(int i = 0; i < scanLineSize; i+=blockSize)
    {
        uchar4 a = globalOffset + ai < N ? g_data[globalOffset + ai] : make_uchar4(0,0,0,0);

        //nextElem = globalOffset + ai + blockSize < N ? g_data[globalOffset + ai + blockSize] : make_uchar4(0,0,0,0);//g_data[blockSize + gpos];

        uint partSumA1 = a.y + a.x;
        uint partSumA2 = a.z + partSumA1;

        uint sum0 = __blockScan<blockSize>(shrdMem, partSumA2 + a.w) - (partSumA2 + a.w) + gsum;

        if(globalOffset + ai < N)
        {
            scanned[globalOffset + ai].x = sum0;
            scanned[globalOffset + ai].y = sum0 + a.x;
            scanned[globalOffset + ai].z = sum0 + partSumA1;
            scanned[globalOffset + ai].w = sum0 + partSumA2;
        }

        gsum += shrdMem[blockSize/32-1];

        //a = nextElem;

        ai += blockSize;
    }

    if(threadIdx.x == blockDim.x-1 && sums)
    {
        sums[blockIdx.x] = gsum;
    }
}

template <
    uint blockSize, 
    typename Operator,
    typename T
>
__global__ void __binaryGroupScan(const T* __restrict g_data, uint* scanned, uint* sums, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    uint elem = op.GetNeutral();
    if(gpos < N)
    {
        elem = op(g_data[gpos]);
    }

    uint sum = __blockBinaryPrefixSums(shrdMem, elem);

    if(gpos < N)
    {
        scanned[gpos] = sum;
    }

    if(threadIdx.x == blockDim.x-1)
    {
        sums[blockIdx.x] = sum + elem;
    }
}

template <
    uint blockSize, 
    typename Operator,
    typename T
>
__global__ void __groupScan(const T* __restrict g_data, uint* scanned, uint* sums, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    uint elem = op.GetNeutral();
    if(gpos < N)
    {
        elem = op(g_data[gpos]);
    }

    uint sum = __blockScan<blockSize>(shrdMem, elem);

    if(gpos < N)
    {
        scanned[gpos] = sum - elem;
    }

    if(threadIdx.x == blockDim.x-1)
    {
        sums[blockIdx.x] = sum;
    }
}

template <
    uint blockSize, 
    typename Operator,
    typename T
>
__global__ void __groupScanOPI(const T* __restrict g_data, uint* scanned, uint* sums, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    uint gpos = blockDim.x * blockIdx.x + threadIdx.x;

    uint elem = op.GetNeutral();
    if(gpos < N)
    {
        elem = op(g_data[gpos], gpos);
    }

    uint sum = __blockScan<blockSize>(shrdMem, elem);

    if(gpos < N)
    {
        scanned[gpos] = sum - elem;
    }

    if(threadIdx.x == blockDim.x-1)
    {
        sums[blockIdx.x] = sum;
    }
}

template <
    uint blockSize, 
    typename Operator, 
    typename T
>
__global__ void __completeBinaryScan(const T* __restrict g_data, uint* scanned, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize];

    __shared__ uint prefixSum;
    prefixSum = 0;

    uint elem = op.GetNeutral();

    if(threadIdx.x < N)
    {
        elem = op(g_data[threadIdx.x]);
    }

    T nextElem = (T)op.GetNeutral();

    for(uint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(blockSize + gpos < N)
        {
            nextElem = g_data[blockSize + gpos];
        }

        uint sum = __blockBinaryPrefixSums(shrdMem, elem);

        if(gpos < N)
        {
            scanned[gpos] = sum + prefixSum;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum + elem;
        }

        elem = op(nextElem);
    }
}

template <
    uint blockSize, 
    typename Operator, 
    typename T
>
__global__ void __completeScan(const T* __restrict g_data, uint* scanned, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    __shared__ uint prefixSum;
    prefixSum = 0;

    uint elem = op.GetNeutral();

    if(threadIdx.x < N)
    {
        elem = op(g_data[threadIdx.x]);
    }

    T nextElem = (T)op.GetNeutral();

    for(uint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(blockSize + gpos < N)
        {
            nextElem = g_data[blockSize + gpos];
        }

        uint sum = __blockScan<blockSize>(shrdMem, elem);

        if(gpos < N)
        {
            scanned[gpos] = sum + prefixSum - elem;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }

        if(blockSize + gpos < N)
        {
            elem = op(nextElem);
        }
    }
}

template <
    uint blockSize, 
    typename Operator, 
    typename T
>
__global__ void __completeScanOPI(const T* __restrict g_data, uint* scanned, Operator op, uint N)
{
    __shared__ uint shrdMem[blockSize/32];

    __shared__ uint prefixSum;
    prefixSum = 0;

    uint elem = op.GetNeutral();

    if(threadIdx.x < N)
    {
        elem = op(g_data[threadIdx.x], threadIdx.x);
    }

    T nextElem = (T)op.GetNeutral();

    for(uint offset = 0; offset < N; offset += blockSize)
    {
        uint gpos = offset + threadIdx.x;

        if(blockSize + gpos < N)
        {
            nextElem = g_data[blockSize + gpos];
        }

        uint sum = __blockScan<blockSize>(shrdMem, elem);

        if(gpos < N)
        {
            scanned[gpos] = sum + prefixSum - elem;
        }

        __syncthreads();

        if(threadIdx.x == blockDim.x-1)
        {
            prefixSum += sum;
        }

        if(blockSize + gpos < N)
        {
            elem = op(nextElem, blockSize + gpos);
        }
    }
}

// __global__ void __spreadScannedSumsSingleT(uint* scanned, const uint* __restrict prefixSum, uint length)
// {
//     uint tileSize = 256 * 8;
//     uint id = tileSize + blockIdx.x * blockDim.x + threadIdx.x;
//     if(id >= length)
//     {
//         return;
//     }
//     scanned[id] += prefixSum[id/tileSize];
// }

__global__ void inline __spreadScannedSumsSingle4T(uint2* scanned, const uint* __restrict prefixSum, uint length, uint scanSize)
{
    const uint elems = 2;
    uint tileSize = scanSize / elems;
    uint id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= length)
    {
        return;
    }
    scanned[tileSize + id].x += prefixSum[(elems*id+0)/(elems*tileSize) + 1];
    scanned[tileSize + id].y += prefixSum[(elems*id+1)/(elems*tileSize) + 1];
//     scanned[tileSize + id].z += prefixSum[(elems*id+2)/(elems*tileSize) + 1];
//     scanned[tileSize + id].w += prefixSum[(elems*id+3)/(elems*tileSize) + 1];
}

template <typename T>
__global__ void __spreadScannedSumsSingle(T* scanned, const T* __restrict prefixSum, uint length)
{
    uint gid = blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= length)
    {
        return;
    }
    scanned[gid] += prefixSum[blockIdx.x+1];
}

namespace nutty
{
    template <
        class T, T neutral
    >
    struct CompactScanOp
    {
        __device__ T operator()(T elem)
        {
            return elem == neutral ? 0 : 1;
        }

        __device__ __host__ T GetNeutral(void)
        {
            return neutral;
        }
    };

    template <
        typename T
    >
    struct PrefixSumOp
    {
        __device__ T operator()(T elem)
        {
            return elem;
        }

        __device__ __host__ T GetNeutral(void)
        {
            return 0;
        }
    };

    namespace cuda
    {
        template <
            typename T,
            typename I,
            typename Operator,
            int WRITE_SUM,
            int IS_INCLUSIVE
        >
        __global__ void _scan(Operator op, const T* content, I* scanned, I* sums, T neutralItem, uint startStage, uint length)
        {
            ShrdMemory<I> s_mem;
            I* shrdMem = s_mem.Ptr();

            uint thid = threadIdx.x;
            uint grpId = blockIdx.x;
            uint globalPos = 2 * (thid + grpId * blockDim.x);

            uint gpos0 = globalPos;
            uint gpos1 = globalPos + 1;

            T i0 = neutralItem;
            T i1 = neutralItem;

            if(gpos0 < length)
            {
                i0 = content[gpos0];
            }

            if(gpos1 < length)
            {
                i1 = content[gpos1];
            }

            //todo
//             uint ai = thid;
//             uint bi = thid + (2 * blockDim.x/2);

            uint bankOffsetA = 0; // CONFLICT_FREE_OFFSET(ai);
            uint bankOffsetB = 0; //CONFLICT_FREE_OFFSET(bi);

            shrdMem[2 * (blockDim.x - thid - 1) + 0 - bankOffsetA] = (I)op(i1); //i1 == neutralItem ? 0 : (neutralItem==-2 ? i1 : 1);
            shrdMem[2 * (blockDim.x - thid - 1) + 1 - bankOffsetB] = (I)op(i0); //i0 == neutralItem ? 0 : (neutralItem==-2 ? i0 : 1);

            I last = op(i1); // == neutralItem ? 0 : 1;

            uint offset = 1;

            __syncthreads();

            for(int i = startStage; i > 1; i >>= 1)
            {
                if(thid < i)
                {
                    int ai = offset*(2*thid);  
                    int bi = offset*(2*thid+1);  
                    
                    //ai += CONFLICT_FREE_OFFSET(ai);
                    //bi += CONFLICT_FREE_OFFSET(bi);

                    if(bi < 2 * blockDim.x)
                    {
                        shrdMem[ai] += shrdMem[bi];
                    }
                }
                offset <<= 1;
                __syncthreads();
            }

            if(thid == 0) 
            { 
                shrdMem[0 /*+ CONFLICT_FREE_OFFSET(2 * blockDim.x)*/] = op(neutralItem);
            }

            __syncthreads();

            for(int i = 1; i <= startStage; i <<= 1)
            {
                if(thid < i)
                {
                    int ai = offset*(2*thid);  
                    int bi = offset*(2*thid+1);

                    //ai += CONFLICT_FREE_OFFSET(ai);
                    //bi += CONFLICT_FREE_OFFSET(bi);

                    if(bi < 2 * blockDim.x)
                    {
                        I t = shrdMem[ai];  
                        shrdMem[ai] += shrdMem[bi]; 
                        shrdMem[bi] = t;
                    }
                }
                offset >>= 1;
                __syncthreads();
            }

            int elemsLeft = 2 * blockDim.x;

            __shared__ I tmp;

            if(IS_INCLUSIVE)
            {
                int dst1 = (2*thid) + 2;
                I s0 = shrdMem[(2*thid) + 0];
                I s1 = shrdMem[(2*thid) + 1];

                __syncthreads();

                if(dst1 < elemsLeft)
                {
                    shrdMem[dst1] = s1;
                }

                shrdMem[(2*thid) + 1] = s0;
                                
                __syncthreads();

                if(gpos1 == length-1 || gpos0 == length-1)
                {
                    tmp = gpos1 == length-1 ? op(i1) : op(i0);
                    shrdMem[0] += gpos1 == length-1 ? op(i1) : op(i0);
                }
                else if(thid == blockDim.x - 1)
                {
                    tmp = op(i1);
                    shrdMem[0] += op(i1);
                }

                __syncthreads();
            }

            if(gpos0 < length)
            {
                scanned[gpos0] = shrdMem[elemsLeft - 1 - (2*thid)];
            }

            if(gpos1 < length)
            {
                scanned[gpos1] = shrdMem[elemsLeft - 1 - (2*thid+1)];
            }

            if(WRITE_SUM && thid == blockDim.x-1)
            {
                I _t = shrdMem[0];
                if(IS_INCLUSIVE)
                {
                    sums[blockIdx.x] = _t - tmp;
                }
                else
                {
                    sums[blockIdx.x] = _t + last;
                }
            }
        }

        template <
            typename T,
            typename ST,
            typename STM
        >
        __device__ void _compact(T* compactedContent, const T* content, const STM* mask, const ST* scanned, ST neutral, uint N)
        {
            uint id = blockIdx.x * blockDim.x + threadIdx.x;

            if(id >= N)
            {
                return;
            }

            STM t = mask[id];

            if(t != neutral)
            {
                t = scanned[id];
                compactedContent[t] = content[id];
            }
        }

        template <
            typename T,
            typename ST,
            typename STM
        >
        __global__ void compact(T* dstContent, T* content, STM* mask, ST* scanned, ST neutral, uint N)
        {
            _compact(dstContent, content, mask, scanned, neutral, N);
        }

        template <
            typename T
        >
        __global__ void spreadSums(T* scanned, const T* prefixSum, uint length)
        {
            uint thid = threadIdx.x;
            uint grpId = blockIdx.x;
            uint N = blockDim.x;
            uint gid = N + grpId * N + thid;

            if(gid >= length)
            {
                return;
            }

            scanned[gid] = scanned[gid] + prefixSum[grpId+1];
        }

//         template <
//             typename T,
//             int steps
//         >
//         __global__ void shiftMemory(T* memory, uint length)
//         {
//             uint id = threadIdx.x + blockIdx.x * blockDim.x;
// 
//             extern __shared__ T s_data[];
// 
//             if(id >= length)
//             {
//                 return;
//             }
// 
//             s_data[threadIdx.x] = memory[id];
// 
//             __syncthreads();
// 
//             if(steps < 0 && (int)threadIdx.x + steps <= 0)
//             {
//                 return;
//             }
// 
//             if(steps > 0 && (int)threadIdx.x + steps >= blockDim.x)
//             {
//                 return;
//             }
// 
//             memory[id + steps] = s_data[threadIdx.x];
//         }

        template <
            typename T,
            typename ST,
            typename STM
        >
        __host__ void Compact(T* dst, T* src, STM* mask, ST* scanned, ST neutralElement, size_t d)
        {
            dim3 grid = cuda::GetCudaGrid(d, (size_t)256);
            compact<<<grid.x, 256, 0, 0>>>(dst, src, mask, scanned, neutralElement, d);
        }

        const static size_t SCAN_ELEMS_PER_BLOCK = 512;

        template <
            typename T,
            typename I
        >
        __host__ void ExclusivePrefixSumScan(T* begin, I* prefixSum, I* sums, I* scannedSums, size_t d)
        {
           PrefixSumOp<T> op;
           _ExclusiveScan(begin, prefixSum, sums, scannedSums, d, op);
        }

        template <
            typename T,
            typename I,
            typename Operation
        >
        __host__  void ScanPerBlock(T* begin, I* scanned, I* sums, size_t d, Operation op, char inclusive, size_t* _grid = NULL, cudaStream_t pStream = NULL)
        {
            size_t elementsPerBlock = SCAN_ELEMS_PER_BLOCK;

            if(elementsPerBlock >= d)
            {
                elementsPerBlock = (uint)d + (d % 2);
            }

            dim3 block = elementsPerBlock / 2;
            dim3 grid = cuda::GetCudaGrid(d, elementsPerBlock);
            
            if(_grid)
            {
                *_grid = grid.x;
            }

            size_t startStage = elementsPerBlock / 2;

            if(!Ispow2(startStage))
            {
                startStage = 1ULL << (GetMSB(startStage) + 1);
            }

            if(inclusive)
            {
                if(grid.x == 1 || !sums)
                {
                    _scan<T, I, Operation, 0, 1><<<grid, block, elementsPerBlock * sizeof(I), pStream>>>(op, begin, scanned, sums, op.GetNeutral(), startStage, d);
                }
                else
                {
                    _scan<T, I, Operation, 1, 1><<<grid, block, elementsPerBlock * sizeof(I), pStream>>>(op, begin, scanned, sums, op.GetNeutral(), startStage, d);
                }
            }
            else
            {
                if(grid.x == 1 || !sums)
                {
                    _scan<T, I, Operation, 0, 0><<<grid, block, elementsPerBlock * sizeof(I), pStream>>>(op, begin, scanned, sums, op.GetNeutral(), startStage, d);
                }
                else
                {
                    _scan<T, I, Operation, 1, 0><<<grid, block, elementsPerBlock * sizeof(I), pStream>>>(op, begin, scanned, sums, op.GetNeutral(), startStage, d);
                }
            }
        }

        template <
            typename T,
            typename I,
            typename Operation,
            int TYPE
        >
       __host__  void __Scan(T* begin, I* scanned, I* sums, I* scannedSums, size_t d, Operation op, cudaStream_t pStream = NULL)
        {
            size_t grid;
            ScanPerBlock(begin, scanned, sums, d, op, TYPE, &grid, pStream); 

            if(grid == 1 || !scannedSums)
            {
                return;
            }

            dim3 sumBlock = (grid + (grid % 2)) / 2;

            size_t startStage = sumBlock.x;

            if(!Ispow2(startStage))
            {
                startStage = 1ULL << (GetMSB(startStage) + 1);
            }
            
            PrefixSumOp<I> _op;
            _scan<I, I, PrefixSumOp<I>, 0, 0><<<1, sumBlock, 2 * sumBlock.x * sizeof(I), pStream>>>(_op, sums, scannedSums, (I*)NULL, _op.GetNeutral(), startStage, grid);
        }

        template <
            typename T
        >
        __host__ void _spreadSums(T* prefixSum, const T* scannedSums, size_t grid, size_t N, cudaStream_t pStream = NULL, uint block = SCAN_ELEMS_PER_BLOCK)
        {
            spreadSums<<<grid, SCAN_ELEMS_PER_BLOCK, 0, pStream>>>(prefixSum, scannedSums, N);
        }

        template <
            typename T,
            typename I,
            typename Operator,
            int TYPE
        >
        __host__ void _ScanComplete(T* begin, I* prefixSum, I* sums, I* scannedSums, size_t d, Operator op, cudaStream_t pStream = NULL)
        {            
            dim3 grid = cuda::GetCudaGrid(d, SCAN_ELEMS_PER_BLOCK);

            if(grid.x == 1)
            {
                __Scan<T, I, Operator, TYPE>(begin, prefixSum, sums, (I*)NULL, d, op, pStream);
                return;
            }

            __Scan<T, I, Operator, TYPE>(begin, prefixSum, sums, scannedSums, d, op, pStream);

            grid.x = grid.x - 1;

            assert(grid.x > 0);

            //spreadSums<<<grid, ELEMS_PER_BLOCK, 0, 0>>>(prefixSum, scannedSums, d);
            _spreadSums(prefixSum, scannedSums, grid.x, d, pStream);
        }

        template <
            typename T,
            typename I,
            typename Operator
        >
        __host__ void _ExclusiveScan(T* begin, I* prefixSum, I* sums, I* scannedSums, size_t d, Operator op, cudaStream_t pStream = NULL)
        {            
            _ScanComplete<T, I, Operator, 0>(begin, prefixSum, sums, scannedSums, d, op, pStream);
        }

        template <
            typename T,
            typename I,
            typename Operator
        >
        __host__ void _InclusiveScan(T* begin, I* prefixSum, I* sums, I* scannedSums, size_t d, Operator op)
        {            
            _ScanComplete<T, I, Operator, 1>(begin, prefixSum, sums, scannedSums, d, op);
        }
    }
}