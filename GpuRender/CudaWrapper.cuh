#pragma once

#include "cuda_runtime.h"

namespace CudaWrapper
{
    void* CudaMalloc(size_t size);
    void CudaMemset(void* memAddress, int value, size_t numBytes);
    void CudaSetDevice(int deviceNum);
    void CudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
    void CudaCheckLaunchErrors();
    void CudaDeviceSynchronize();
}