#pragma once
#include "base/scan.h"
#include "Copy.h"
//#include <sm_35_intrinsics.h>
namespace nutty
{
    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator,
        typename Operator
    >
    void InclusiveScan(Iterator_& begin, Iterator_& end, ScanIterator& prefixSum, SumIterator& sums, SumIterator& scannedSums, Operator op)
    {
        nutty::cuda::_InclusiveScan(begin(), prefixSum(), sums(), scannedSums(), Distance(begin, end), op);
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator,
        typename Operator
    >
    void ExclusiveScan(Iterator_& begin, Iterator_& end, ScanIterator& scanned, SumIterator& sums, SumIterator& scannedSums, Operator op, cudaStream_t pStream = NULL)
    { 
        nutty::cuda::_ExclusiveScan(begin(), scanned(), sums(), scannedSums(), Distance(begin, end), op, pStream);  
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename SumIterator
    >
    void ExclusivePrefixSumScan(Iterator_& begin, Iterator_& end, ScanIterator& prefixSum, SumIterator& sums, SumIterator& scannedSums)
    {
        nutty::cuda::ExclusivePrefixSumScan(begin(), prefixSum(), sums(), scannedSums(), Distance(begin, end));
    }

    template <
        typename Iterator_,
        typename ScanIterator,
        typename MaskIterator,
        typename T
    >
    void Compact(Iterator_& dstBegin, Iterator_& begin, Iterator_& end, MaskIterator& mask, ScanIterator& dstAddress, T neutral)
    {
        nutty::cuda::Compact(dstBegin(), begin(), mask(), dstAddress(), neutral, Distance(begin, end));
    }

    template <
        typename Iterator_,
        typename MaskIterator_
    >
    void MakeExclusive(Iterator_& dstBegin, Iterator_& srcBegin, Iterator_& srcEnd, MaskIterator_& mask)
    {
        static const uint zzerro = 0;
        nutty::Copy(dstBegin, zzerro);
        nutty::Copy(dstBegin + 1, srcBegin, srcEnd - 1);
        uint endVal = 0;
        size_t end = nutty::Distance(srcBegin, srcEnd) - 1;
        endVal = *(srcEnd - 1) - *(mask + end);
        nutty::Copy(dstBegin + end, endVal);
    }

    class Scanner
    {
    public:
        nutty::DeviceBuffer<uint> m_scannedSums;
        nutty::DeviceBuffer<uint> m_sums;
    private:
        nutty::DeviceBuffer<uint> m_scannedData;
        //nutty::DeviceBuffer<uint> m_scannedSums;
        Scanner* m_pSumScanner;

    public:
        Scanner(void) : m_pSumScanner(NULL)
        {

        }

        void Resize(size_t size)
        {
            if(size <= m_scannedData.Size())
            {
                return;
            }

            m_scannedData.Resize(size);
#if 0
            size_t sumSize = (size % nutty::cuda::SCAN_ELEMS_PER_BLOCK) == 0 ? size / nutty::cuda::SCAN_ELEMS_PER_BLOCK : (size / nutty::cuda::SCAN_ELEMS_PER_BLOCK) + 1;
#else
            size_t sumSize = size / 256 + 1;//(size % 256) == 0 ? size / 256 : (size / 256) + 1;
#endif

            m_sums.Resize(sumSize);
            m_scannedSums.Resize(sumSize);

//             if(sumSize > 2048) //doesn't fit in one workgroup
//             {
//                 m_pSumScanner = new Scanner();
// 
//                 m_pSumScanner->Resize(sumSize);
//             }
        }

        template<
            typename Iterator,
            typename Operator
        >
        void IncScan(Iterator& begin, Iterator& end, Operator op)
        {
            //nutty::InclusiveScan(begin, end, m_scannedData.Begin(), m_sums.Begin(), m_scannedSums.Begin(), op);
        }

        template<
            typename Iterator,
            typename Operator
        >
        void ExcBinaryScan(Iterator& begin, Iterator& end, Operator op, cudaStream_t pStream = NULL)
        {
            const static uint BLOCK_SIZE = 512U;
            uint N = (uint)nutty::Distance(begin, end);

            if(N < 768)
            {
                __completeBinaryScan<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
            }
            else
            {
                uint grid = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                __binaryGroupScan<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, pStream>>>(
                    begin(), m_scannedData.GetPointer(), m_sums.GetPointer(), op, N);

                uint sumsCount = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                nutty::PrefixSumOp<uint> _op;
                __completeScan<1024><<<1, 1024, 0, pStream>>>(m_sums.GetConstPointer(), m_sums.GetPointer(), _op, sumsCount);

                __spreadScannedSumsSingle<<<grid-1, BLOCK_SIZE, 0, pStream>>>(m_scannedData.GetPointer(), m_sums.GetPointer(), N);
            }
        }

        template<
            typename Iterator,
            typename Operator
        >
        void ExcScanOPI(Iterator& begin, Iterator& end, Operator op, cudaStream_t pStream = NULL)
        {
            const static uint BLOCK_SIZE = 256U;
            uint N = (uint)nutty::Distance(begin, end);
            if(N < 2 * BLOCK_SIZE)
            {
                if(N > 256)
                {
                    __completeScanOPI<256><<<1, 256, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                } 
                else if(N > 128)
                {
                    __completeScanOPI<128><<<1, 128, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);

                } 
                else if(N > 64)
                {
                    __completeScanOPI<64><<<1, 64, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                }
                else
                {
                    __completeScanOPI<32><<<1, 32, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                }
            }
            else
            {
                uint grid = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                __groupScanOPI<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, pStream>>>(
                    begin(), m_scannedData.GetPointer(), m_sums.GetPointer(), op, N);

                uint sumsCount = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                nutty::PrefixSumOp<uint> _op;
                __completeScan<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, pStream>>>(m_sums.GetConstPointer(), m_sums.GetPointer(), _op, sumsCount);

                __spreadScannedSumsSingle<<<grid-1, BLOCK_SIZE, 0, pStream>>>(m_scannedData.GetPointer(), m_sums.GetPointer(), N);
            }
        }

        template<
            typename Iterator,
            typename Operator
        >
        void ExcScan(Iterator& begin, Iterator& end, Operator op, cudaStream_t pStream = NULL)
        {
            const static uint BLOCK_SIZE = 256U;
            uint N = (uint)nutty::Distance(begin, end);
            if(N < 2 * BLOCK_SIZE)
            {
                if(N > 256)
                {
                    __completeScan<256><<<1, 256, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                } 
                else if(N > 128)
                {
                    __completeScan<128><<<1, 128, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);

                } 
                else if(N > 64)
                {
                    __completeScan<64><<<1, 64, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                }
                else
                {
                    __completeScan<32><<<1, 32, 0, pStream>>>(begin(), m_scannedData.GetPointer(), op, N);
                }
            }
            else
            {
                uint grid = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                __groupScan<BLOCK_SIZE><<<grid, BLOCK_SIZE, 0, pStream>>>(
                    begin(), m_scannedData.GetPointer(), m_sums.GetPointer(), op, N);

                uint sumsCount = nutty::cuda::GetCudaGrid(N, BLOCK_SIZE);
                nutty::PrefixSumOp<uint> _op;
                __completeScan<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, pStream>>>(m_sums.GetConstPointer(), m_sums.GetPointer(), _op, sumsCount);

                __spreadScannedSumsSingle<<<grid-1, BLOCK_SIZE, 0, pStream>>>(m_scannedData.GetPointer(), m_sums.GetPointer(), N);
            }
        }

        nutty::DeviceBuffer<uint>& GetPrefixSum(void)
        {
            return m_scannedData;
        }

        ~Scanner(void)
        {
            if(m_pSumScanner)
            {
                delete m_pSumScanner;
                m_pSumScanner = NULL;
            }
        }
    };

    template <typename T, typename SumOperator>
    class TScanner
    {
    private:
        nutty::DeviceBuffer<T> m_scannedData;
        nutty::DeviceBuffer<T> m_sums;
        nutty::DeviceBuffer<T> m_scannedSums;
        TScanner<T, SumOperator>* m_pSumScanner;

    public:
        TScanner(void) : m_pSumScanner(NULL)
        {

        }

        void Resize(size_t size)
        {
            if(size <= m_scannedData.Size())
            {
                return;
            }
            m_scannedData.Resize(size);

            size_t sumSize = (size % nutty::cuda::SCAN_ELEMS_PER_BLOCK) == 0 ? size / nutty::cuda::SCAN_ELEMS_PER_BLOCK : (size / nutty::cuda::SCAN_ELEMS_PER_BLOCK) + 1;

            m_sums.Resize(sumSize);
            m_scannedSums.Resize(sumSize);

            if(sumSize > 2048) //doesn't fit in one workgroup
            {
                m_pSumScanner = new TScanner<T, SumOperator>();

                m_pSumScanner->Resize(sumSize);
            }
        }

        template<
            typename Iterator,
            typename Operator
        >
        void ExcScan(Iterator& begin, Iterator& end, Operator op)
        {
            if(m_pSumScanner)
            {
                size_t d = nutty::Distance(begin, end);
                size_t grid;
                nutty::cuda::ScanPerBlock(begin(), m_scannedData.Begin()(), m_sums.Begin()(), d, op, 0, &grid);
                SumOperator _op;
                m_pSumScanner->ExcScan(m_sums.Begin(), m_sums.End(), _op);
                nutty::cuda::_spreadSums(m_scannedData.Begin()(), m_pSumScanner->GetPrefixSum().Begin()(), grid - 1, m_scannedData.Size());
            }
            else
            {
                size_t d = nutty::Distance(begin, end);
                size_t grid;
                nutty::cuda::ScanPerBlock(begin(), m_scannedData.Begin()(), m_sums.Begin()(), d, op, 0, &grid);
                SumOperator _op;
                nutty::cuda::ScanPerBlock(m_sums.Begin()(), m_scannedSums.Begin()(), (T*)NULL, m_sums.Size(), _op, 0);//m_sums.Size());
                nutty::cuda::_spreadSums(m_scannedData.Begin()(), m_scannedSums.Begin()(), grid - 1, m_scannedData.Size());
            }
        }

        const nutty::DeviceBuffer<T>& GetPrefixSum(void)
        {
            return m_scannedData;
        }

        ~TScanner(void)
        {
            if(m_pSumScanner)
            {
                delete m_pSumScanner;
                m_pSumScanner = NULL;
            }
        }
    };
}