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
    unsigned char value = 0;
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
OpacityResult PixelOpacity(Point pt, Line line)
{
    float distSquared = line.DistSquaredTo(pt);
    if (abs(distSquared) <= 1)
    {
        auto alpha = round(255 * (1 - abs(distSquared)));
        return OpacityResult(alpha);
    }
    return OpacityResult();
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

__device__
Pixel RenderPixel(Point pt, Pixel originalColor, Pixel color, Line* lineArray, int numLines)
{
    OpacityResult maxValue;
    for (int i = 0; i < numLines; i++)
    {
        Line line = lineArray[i];
        auto results = PixelOpacity(pt, line);

        if (results.isvalid && results.value >= maxValue.value)
        {
            maxValue = results;
        }
    }
    
    if (maxValue.isvalid)
    {
        color.A = maxValue.value;
        return BlendPixels(originalColor, color);
    }
    else
    {
        return originalColor;
    }
}

__global__
void RenderKernel(Pixel* pixels, int imgWidth, int imgHeight, Pixel color, Line* lineArray, int numLines)
{
    __shared__ Pixel opacity;
    int pixelNum = blockIdx.x;
    int lineNum = threadIdx.x;
    int x = pixelNum % imgWidth;
    int y = pixelNum / imgWidth;
    
    if (x > imgWidth || y > imgHeight || lineNum > numLines)
        return;

    Point pt;
    pt.x = x;
    pt.y = y;

    Line line = lineArray[lineNum];
    auto results = PixelOpacity(pt, line);

    if (results.isvalid)
    {
        color.A = results.value;
        //auto address = &opacities[pixelNum];
        atomicAlphaMax(&opacity, color);
    }

    __syncthreads();
    if (lineNum == 0)
    {
        pixels[pixelNum] = BlendPixels(pixels[pixelNum], opacity);
    }
}

__global__
void FlattenImages(Pixel* base, Pixel* overlay, int imgWidth, int imgHeight)
{
    int pixelNum = threadIdx.x + blockIdx.x * blockDim.x;
    int x = pixelNum % imgWidth;
    int y = pixelNum / imgWidth;

    if (x > imgWidth || y > imgHeight)
        return;

    base[pixelNum] = BlendPixels(base[pixelNum], overlay[pixelNum]);
}

void CudaRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
    Pixel* cudaImg = nullptr;
    Line* cudaLines = nullptr;
    //Pixel* cudaOpacities = nullptr;
    try
    {
        CudaSetDevice(0);

        // allocate memory
        int numPixels = image->width * image->height;
        int numWorkers = numPixels * lines->size();
        
        cudaImg = (Pixel*)CudaMalloc(numPixels * sizeof(Pixel));
        CudaMemcpy(cudaImg, image->pixels, numPixels * sizeof(Pixel), cudaMemcpyHostToDevice);

        cudaLines = (Line*)CudaMalloc(lines->size() * sizeof(Line));
        CudaMemcpy(cudaLines, lines->data(), lines->size() * sizeof(Line), cudaMemcpyHostToDevice);

        //cudaOpacities = (Pixel*)CudaMalloc(numPixels * sizeof(Pixel));
        //CudaMemset(cudaOpacities, 0, numPixels * sizeof(Pixel));
        
        // Figure out how many threads and blocks
        int numBlocks = numWorkers / THREADS_PER_BLOCK;
        int numThreads = THREADS_PER_BLOCK;

        printf("Number of blocks: %d", numBlocks);

        // launch cuda kernal and wait for it to finish
        RenderKernel<<<numPixels, lines->size()>>>(cudaImg, image->width, image->height, color, cudaLines, lines->size());
        CudaCheckLaunchErrors();
        CudaDeviceSynchronize();

        //numBlocks = numPixels / THREADS_PER_BLOCK;
        //FlattenImages<<<numBlocks, numThreads>>>(cudaImg, cudaOpacities, image->width, image->height);
        //CudaCheckLaunchErrors();
        //CudaDeviceSynchronize();

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
    if (cudaLines != nullptr)
    {
        cudaFree(cudaLines);
    }
    //if (cudaOpacities != nullptr)
    //{
    //    cudaFree(cudaOpacities);
    //}
    cudaDeviceReset();
}