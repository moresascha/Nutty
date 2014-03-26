#pragma once
#include "../Inc.h"
#include <fstream>
namespace nutty
{
    struct GlobalPtr
    {
        CUdeviceptr ptr;
        size_t byteSize;
    };


    class cuTexRef
    {
    private:
        CUtexref m_ref;

    public:
        cuTexRef(void)
        {

        }

        cuTexRef(CUtexref ref) : m_ref(ref)
        {

        }

        cuTexRef(const cuTexRef& ref)
        {
            m_ref = ref.m_ref;
        }

        void Create(CUtexref ref)
        {
            m_ref = ref;
        }

        void NormalizedCoords(void)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFlags(m_ref, CU_TRSF_NORMALIZED_COORDINATES));
        }

        void SetFlags(uint flags)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFlags(m_ref, flags));
        }

        CUdeviceptr GetAddress(void)
        {
            CUdeviceptr ptr;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefGetAddress(&ptr, m_ref));
            return ptr;
        }

        CUarray GetArray(void)
        {
            CUarray ptr;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefGetArray(&ptr, m_ref));
            return ptr;
        }

        void BindToArray(CUarray hArray)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetArray(m_ref, hArray, CU_TRSA_OVERRIDE_FORMAT));
        }

        void SetFilterMode(CUfilter_mode mode)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFilterMode(m_ref, mode));
        }

        void SetFormat(CUarray_format fmt, int cmpCount)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFormat(m_ref, fmt, cmpCount));
        }

        void SetAddressMode(CUaddress_mode mode, int dim)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetAddressMode(m_ref, dim, mode));
        }
    };

    class cuModule
    {
    private:
        std::string m_file;
        CUmodule m_cudaModule;

    public:
        cuModule(const char* file) : m_cudaModule(NULL)
        {
            Create(file);
        }

        cuModule(void) : m_cudaModule(NULL)
        {
        }

        void Create(const char* file = NULL)
        {
            if(file)
            {
                m_file = file;
            }
            if(m_cudaModule)
            {
                _Delete();
            }
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleLoad(&m_cudaModule, m_file.c_str()));
        }

        CUfunction GetFunction(const char* name)
        {
            CUfunction func;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetFunction(&func, m_cudaModule, name));
            return func;
        }

        cuTexRef GetTexRef(const char* name)
        {
            CUtexref ref;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetTexRef(&ref, m_cudaModule, name));
            cuTexRef cuRef(ref);
            return cuRef;
        }

        GlobalPtr GetGlobal(const char* name)
        {
            GlobalPtr ptr;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetGlobal(&ptr.ptr,  &ptr.byteSize, m_cudaModule, name));
            return ptr;
        }

        void _Delete(void)
        {
            CHECK_CNTX();
            if(m_cudaModule)
            {
                CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleUnload(m_cudaModule));
            }
        }

        ~cuModule(void)
        {
            _Delete();
        }
    };
}