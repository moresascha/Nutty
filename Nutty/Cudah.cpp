#include "Inc.h"
#include "cudah.h"

#include "CUBuffer.h"
#include "CUHelper.h"
#include "CUModule.h"
#include "CUKernel.h"

#define _CUDA_ALONE 1

namespace cudah 
{

    //INT cudah::m_sDev = 0;
    //cudaDeviceProp cudah::m_sProps = cudaDeviceProp();

    /*
	VOID _cuda_array::VDestroy(VOID)
    {
        if(m_array)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuArrayDestroy(m_array));
        }
    }

    template<typename T>
    void copyLinearToPitchedMemory(T* devPtr, CONST T* linearData, UINT pitch, UINT width, UINT height, UINT depth)
    {
        UINT slicePitch = pitch * height;
        UINT index = 0;
        for (UINT z = 0; z < depth; ++z) 
        {
            T* slice = devPtr + z * slicePitch;

            for (UINT y = 0; y < height; ++y) 
            {
                T* row = slice + y * pitch;

                for (UINT x = 0; x < width; ++x) 
                {
                    row[x] = linearData[index++];
                }
            }
        }
    }

    cudah::cudah(LPCSTR file)
    { 
        std::string path;
#if !defined(_CUDA_ALONE)
        path += app::g_pApp->GetConfig()->GetString("sPTXPath");
#endif
        m_module.m_file = path + file;
        OnRestore();

#if defined(_DEBUG) && !defined(_CUDA_ALONE)
        std::vector<std::string> p = util::split(std::string(file), '/');
        std::string& s = p[p.size()-1];
        std::string fn = util::split(s, '.')[0];
        fn += ".cu";
        std::wstring finalName(fn.begin(), fn.end());
        m_modProc = 
            std::shared_ptr<proc::WatchCudaFileModificationProcess>(new proc::WatchCudaFileModificationProcess(this, finalName.c_str(), L"./chimera/"));
        app::g_pApp->GetLogic()->AttachProcess(m_modProc);
#endif
    }

    CUcontext cudah::GetContext(VOID)
    {
        return g_CudaContext;
    }

    CUdevice cudah::GetDevice(VOID)
    {
        return g_device;
    }

    cudah::cudah(VOID)
    {
    }

    VOID cudah::OnRestore(VOID)
    {
        if(m_module.m_cudaModule)
        {
            cuModuleUnload(m_module.m_cudaModule);
        }

		std::ifstream f;
		f.open(m_module.m_file.c_str());

		if(!f.good())
		{
			LOG_CRITICAL_ERROR_A("File '%s' not found.", m_module.m_file.c_str());
		}

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleLoad(&m_module.m_cudaModule, m_module.m_file.c_str()));

        TBD_FOR(m_module.m_kernel)
        {
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetFunction(&it->second->m_fpCuda, m_module.m_cudaModule, it->second->m_func_name.c_str()));
        }

        std::vector<TextureBindung> bindung = m_textureBinding;
        m_textureBinding.clear();
        TBD_FOR(bindung)
        {
            BindArrayToTexture(it->m_textureName.c_str(), it->m_cudaArray, it->m_filterMode, it->m_addressMode, it->m_flags);
        }
    }

    cuda_kernel cudah::GetKernel(LPCSTR name)
    {
        auto it = m_module.m_kernel.find(name);

        if(it != m_module.m_kernel.end())
        {
            return it->second;
        }

        cuda_kernel kernel = new _cuda_kernel();

        kernel->m_func_name = std::string(name);

        kernel->m_stream = g_defaultStream;

        assert(m_module.m_cudaModule != NULL);

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetFunction(&kernel->m_fpCuda, m_module.m_cudaModule, name));

        m_module.m_kernel[std::string(name)] = kernel;

        return kernel;
    }

    GlobalPtr cudah::GetGlobal(LPCSTR global)
    {
        GlobalPtr ptr;
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetGlobal(&ptr.ptr,  &ptr.byteSize, m_module.m_cudaModule, global));
        return ptr;
    }

    VOID cudah::BindArrayToTexture(LPCSTR textur, cuda_array array, CUfilter_mode filterMode, CUaddress_mode addressMode, UINT flags)
    {
        CUtexref ref;
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuModuleGetTexRef(&ref, m_module.m_cudaModule, textur));

        TextureBindung bind;
        bind.m_addressMode = addressMode;
        bind.m_cudaArray = array;
        bind.m_filterMode = filterMode;
        bind.m_flags = flags;
        bind.m_textureName = std::string(textur);
        m_textureBinding.push_back(bind);

        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetArray(ref, array->m_array, CU_TRSA_OVERRIDE_FORMAT));
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFlags(ref, flags));
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetFilterMode(ref, filterMode));
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuTexRefSetAddressMode(ref, 0, addressMode));
    }

    cuda_array cudah::CreateArray(CONST std::string& name, CONST UINT w, CONST UINT h, CONST UINT d, ArrayFormat format, FLOAT* data)
    {
        CUarray a;

        CUDA_ARRAY3D_DESCRIPTOR desc;
        desc.Format = CU_AD_FORMAT_FLOAT;
        desc.NumChannels = 1 + format;

        //cudaExtent ex = make_cudaExtent(w, h, d); 
        desc.Width = w;
        desc.Height = h;
        desc.Depth = d;
        desc.Flags = 0;
        
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuArray3DCreate(&a, &desc));

        //CUDA_RUNTIME_SAFE_CALLING(cudaMalloc3DArray(&array, &desc, ex)); //TODO: flag

        cuda_array c_array = new _cuda_array(a);
        //c_array->desc = desc;
        m_array[name] = c_array;

        if(data)
        {
            CUDA_MEMCPY3D cpy;
            ZeroMemory(&cpy, sizeof(CUDA_MEMCPY3D));
            cpy.Depth = d;
            cpy.WidthInBytes = w * desc.NumChannels * sizeof(FLOAT);
            cpy.Height = h;
            cpy.srcHeight = h;
            cpy.srcPitch = cpy.WidthInBytes;
            cpy.dstArray = a;
            cpy.srcHost = data;
            cpy.srcMemoryType = CU_MEMORYTYPE_HOST;
            cpy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
            //cpy.
            CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuMemcpy3D(&cpy));
            //SAFE_ARRAY_DELETE(pitched);
        }

        return c_array;
    }

    cuBuffer cudah::RegisterD3D11Buffer(std::string& name, ID3D11Resource* resource, enum cudaGraphicsMapFlags flags)
    {

#if defined CUDA_SAFE
        CheckResourceNotExists(name);
#endif
        cuBuffer b = new _cuda_buffer;
        b->isFromGAPI = TRUE;
        
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuGraphicsD3D11RegisterResource(&b->glRes, resource, flags));

        m_buffer.insert(std::pair<std::string, cuBuffer>(name, b));
        return b;
    }

    map_info cudah::MapGraphicsResource(cuBuffer buffer) {

#if defined CUDA_SAFE
        if(!buffer->isFromGAPI) {
            return map_info();
        }
#endif
        //CUDA_RUNTIME_SAFE_CALLING(cudaGraphicsMapResources(1, &buffer->glRes, 0));
        
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuGraphicsMapResources(1, &buffer->glRes, 0));
        map_info info;
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuGraphicsResourceGetMappedPointer(&(info.ptr), &(info.size), buffer->glRes));

        buffer->ptr = info.ptr;
        return info;
    }

    VOID cudah::UnmapGraphicsResource(cuBuffer buffer) 
    {
        CUDA_DRIVER_SAFE_CALLING_NO_SYNC(cuGraphicsUnmapResources(1, &buffer->glRes, 0));
    }
    
    */
}
