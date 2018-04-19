#include "cuda_runtime.h"
#include <vector>

#include "Image.cuh"
#include "CudaWrapper.cuh"
#include "CudaRender.cuh"

const int THREADS_PER_BLOCK = 1024;

using namespace CudaWrapper;

struct OpacityResult
{
    __device__
    OpacityResult() {}

    __device__
    OpacityResult(unsigned char value)
    {
        this->value = value;
        isvalid = true;
    }

    bool isvalid = false;
    uint8_t value = 0;
};

__device__
Pixel BlendPixels(Pixel p1, Pixel p2)
{
    Pixel newPixel;
    newPixel.R = (p1.R * (255 - p2.A) + p2.R * p2.A) / 255;
    newPixel.G = (p1.G * (255 - p2.A) + p2.G * p2.A) / 255;
    newPixel.B = (p1.B * (255 - p2.A) + p2.B * p2.A) / 255;
    newPixel.A = p1.A + p2.A*(255 - p1.A);
    return newPixel;
}

__device__
Pixel atomicAlphaMax(Pixel* address, Pixel value)
{
    unsigned int* addrAsUint = (unsigned int*)address;
    unsigned int old = *addrAsUint;
    unsigned int assumed;

    do
    {
        assumed = old;
        Pixel* tmp = (Pixel*)&assumed;
        if (value.A > tmp->A)
        {
            old = atomicCAS(addrAsUint, assumed, *(unsigned int*)&value);
        }
        else
        {
            break;
        }
    } while (assumed != old);
    return *((Pixel*)&old);
}

union IntBytes
{
    int intVal;
    uint8_t bytes[4];
};

__device__
uint8_t atomicMax(uint8_t* address, uint8_t value)
{
    int* fullAddress = (int*)((size_t)address & ~0x3);
    int offset = (size_t)address & 0x3;

    while (true)
    {
        // Get the current value
        IntBytes curVal;
        curVal.intVal = *fullAddress;

        // If the value currently in the memory address is larger than the new value, discard the new value
        if (curVal.bytes[offset] >= value)
        {
            return curVal.bytes[offset];
        }

        IntBytes newVal = curVal;
        newVal.bytes[offset] = value;
        
        // Place the new value into memory if nobody else has changed 
        //   the memory address since we pulled the value
        int oldVal = atomicCAS(fullAddress, curVal.intVal, newVal.intVal);

        // If the swap was successful (nobody else changed the memory location)
        // then we're done
        // otherwise repeat the entire process
        if (curVal.intVal == oldVal)
        {
            return curVal.bytes[offset];
        }
    }
}

__device__
void DrawPixelsAt(uint8_t* opacities, int imgWidth, int imgHeight, Line line, int x, int y, float distFromPixel)
{
    if (x > imgWidth || y > imgHeight)
        return;

    uint8_t val = (1 - abs(distFromPixel)) * 255;
    if (val > 0)
    {
        atomicMax(&opacities[x + y * imgWidth], val);
    }
}

__global__
void DrawLineYCentric(uint8_t* opacities, int imgWidth, int imgHeight, Line line)
{
    Point start = line.start;
    Point end = line.end;
    if (line.start.y > line.end.y)
    {
        start = line.end;
        end = line.start;
    }

    int dx = end.x - start.x;
    int dy = end.y - start.y;


    int y = threadIdx.x + blockIdx.x * blockDim.x;
    if (y > abs(line.Vector().y))
        return;
    y += start.y;

    float x = start.x + dx / (float)dy * (y - start.y);
    if (threadIdx.y == 1)
    {
        DrawPixelsAt(opacities, imgWidth, imgHeight, line, ceil(x), y, x - ceil(x));
    }
    else
    {
        DrawPixelsAt(opacities, imgWidth, imgHeight, line, floor(x), y, x - floor(x));
    }
}

__global__
void DrawLineXCentric(uint8_t* opacities, int imgWidth, int imgHeight, Line line)
{
    Point start = line.start;
    Point end = line.end;
    if (line.start.x > line.end.x)
    {
        start = line.end;
        end = line.start;
    }

    int dx = end.x - start.x;
    int dy = end.y - start.y;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x > abs(line.Vector().x))
        return;
    x += start.x;

    float y = start.y + dy / (float)dx * (x - start.x);
    if (threadIdx.y == 1)
    {
        DrawPixelsAt(opacities, imgWidth, imgHeight, line, x, ceil(y), y - ceil(y));
    }
    else
    {
        DrawPixelsAt(opacities, imgWidth, imgHeight, line, x, floor(y), y - floor(y));
    }
}

__global__
void FlattenImages(Pixel* base, uint8_t* overlayOpacities, Pixel lineColor, int imgWidth, int imgHeight)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pixelNum = x + y * imgWidth;

    if (x > imgWidth || y > imgHeight)
        return;

    lineColor.A = overlayOpacities[pixelNum];
    base[pixelNum] = BlendPixels(base[pixelNum], lineColor);
}

void CudaRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
    Pixel* cudaImg = nullptr;
    uint8_t* cudaOpacities = nullptr;
    try
    {
        CudaSetDevice(0);

        // allocate memory
        int numPixels = image->width * image->height;
                
        cudaStream_t memStream;
        cudaStreamCreate(&memStream);
        cudaImg = (Pixel*)CudaMalloc(numPixels * sizeof(Pixel));
        cudaMemcpyAsync(cudaImg, image->pixels, numPixels * sizeof(Pixel), cudaMemcpyHostToDevice, memStream);

        cudaOpacities = (uint8_t*)CudaMalloc(numPixels * sizeof(uint8_t));
        CudaMemset(cudaOpacities, 0, numPixels * sizeof(uint8_t));
        
        for (int i = 0; i < lines->size(); i++)
        {
            auto line = lines->at(i);
            auto lineVect = line.Vector();

            if (abs(lineVect.x) >= abs(lineVect.y))
            {
                dim3 numThreads(THREADS_PER_BLOCK / 2, 2);
                dim3 numBlocks(abs(lineVect.x) / numThreads.x, 1, 1);
                DrawLineXCentric<<<numBlocks,numThreads>>>(cudaOpacities, image->width, image->height, line);
                CudaCheckLaunchErrors();
            }
            else
            {
                dim3 numThreads(THREADS_PER_BLOCK / 2, 2);
                dim3 numBlocks(abs(lineVect.y) / numThreads.x, 1, 1);
                DrawLineYCentric<<<numBlocks,numThreads>>>(cudaOpacities, image->width, image->height, line);
                CudaCheckLaunchErrors();
            }
        }

        CudaDeviceSynchronize();
        
        int sqrtThreadsPerBlock = sqrt(THREADS_PER_BLOCK);
        dim3 numBlocksFlatten(image->width / sqrtThreadsPerBlock, image->height / sqrtThreadsPerBlock);
        dim3 numThreadsFlatten(sqrtThreadsPerBlock, sqrtThreadsPerBlock);
        FlattenImages<<<numBlocksFlatten, numThreadsFlatten>>>(cudaImg, cudaOpacities, color, image->width, image->height);

        // copy the results into the image
        CudaMemcpy(image->pixels, cudaImg, numPixels * sizeof(Pixel), cudaMemcpyDeviceToHost);
    }
    catch (const std::exception& ex)
    {
        printf(ex.what());
    }

    // Free allocations
    if (cudaImg != nullptr)
    {
        cudaFree(cudaImg);
    }
    if (cudaOpacities != nullptr)
    {
        cudaFree(cudaOpacities);
    }
    cudaDeviceReset();
}