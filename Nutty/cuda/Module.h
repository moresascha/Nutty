#pragma once
#include "Inc.h"
#include <fstream>
namespace nutty
{
    struct GlobalPtr
    {
        CUdeviceptr ptr;
        size_t byteSize;
    };

    class cuModule
    {
    private:
        std::string m_file;
        CUmodule m_cudaModule;

    public:
        cuModule(const char* file) : m_cudaModule(NULL), m_file(file)
        {
/*
            std::ifstream f(file);
            if(!f.good())
            {
                LOG_CRITICAL_ERROR_A("File '%s' not found!", file);
            }*/
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleLoad(&m_cudaModule, m_file.c_str()));
        }

        CUfunction GetFunction(const char* name)
        {
            CUfunction func;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetFunction(&func, m_cudaModule, name));
            return func;
        }

        GlobalPtr GetGlobal(const char* name)
        {
            GlobalPtr ptr;
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetGlobal(&ptr.ptr,  &ptr.byteSize, m_cudaModule, name));
            return ptr;
        }

        ~cuModule(void)
        {
            CHECK_CNTX();
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleUnload(m_cudaModule));
        }
    };
}