#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CudaWrapper.cuh"

#include <exception>
#include <string>

void* CudaWrapper::CudaMalloc(size_t size)
{
    void* mem;
    auto status = cudaMalloc((void**)&mem, size);
    if (status != cudaSuccess)
    {
        std::string str = "cudaMalloc returned an error: ";
        str += cudaGetErrorString(status);
        str += "\n";
        throw new std::exception(str.c_str());
    }
    else
    {
        return mem;
    }
}

void CudaWrapper::CudaMemset(void* memAddress, int value, size_t numBytes)
{
    auto status = cudaMemset(memAddress, value, numBytes);
    if (status != cudaSuccess)
    {
        std::string str = "cudaMemset returned an error: ";
        str += cudaGetErrorString(status);
        str += "\n";
        throw new std::exception(str.c_str());
    }
}

void CudaWrapper::CudaSetDevice(int deviceNum)
{
    auto status = cudaSetDevice(deviceNum);
    if (status != cudaSuccess)
    {
        std::string str = "cudaSetDevice returned an error: ";
        str += cudaGetErrorString(status);
        str += "\n";
        throw new std::exception(str.c_str());
    }
}

void CudaWrapper::CudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind)
{
    auto status = cudaMemcpy(dst, src, count, kind);
    if (status != cudaSuccess)
    {
        std::string str = "cudaMemcpy returned an error: ";
        str += cudaGetErrorString(status);
        str += "\n";
        throw new std::exception(str.c_str());
    }
}

void CudaWrapper::CudaCheckLaunchErrors()
{
    // Check for any errors launching the kernel
    auto cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        std::string str = "addKernel launch failed: ";
        str += cudaGetErrorString(cudaStatus);
        str += "\n";
        throw new std::exception(str.c_str());
    }
}

void CudaWrapper::CudaDeviceSynchronize()
{
    auto cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        std::string str = "cudaDeviceSynchronize returned an error: ";
        str += cudaGetErrorString(cudaStatus);
        str += "\n";
        throw new std::exception(str.c_str());
    }
}