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


__global__
void RenderKernel(Pixel* pixels, int imgWidth, int imgHeight, Pixel color, Line* lineArray, int numLines)
{
    extern __shared__ uint8_t opacities[];
    int lineNum = threadIdx.y;
    int pixelNum = threadIdx.x + blockIdx.x * blockDim.x;;
    int x = pixelNum % imgWidth;
    int y = pixelNum / imgWidth;
    
    if (x > imgWidth || y > imgHeight || lineNum > numLines)
        return;

    Point pt;
    pt.x = x;
    pt.y = y;
    //printf("Point: %d,%d,%d\n", x, y, lineNum);

    Line line = lineArray[lineNum];
    auto results = PixelOpacity(pt, line);

    int opacity = 0;
    if (results.isvalid)
    {
        opacity = results.value;
    }
    opacities[threadIdx.x * numLines + lineNum] = opacity;

    __syncthreads();

    if (lineNum == 0)
    {
        color.A = 0;
        for (int i = 0; i < numLines; i++)
        {
            uint8_t tmpAlpha = opacities[threadIdx.x * numLines + i];
            if (tmpAlpha > color.A)
            {
                color.A = tmpAlpha;
            }
        }
        pixels[pixelNum] = BlendPixels(pixels[pixelNum], color);
    }
}

void CudaRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
    Pixel* cudaImg = nullptr;
    Line* cudaLines = nullptr;
    try
    {
        CudaSetDevice(0);

        // allocate memory
        int numPixels = image->width * image->height;
        int numWorkers = numPixels * lines->size();

        int pixelsPerBlock = THREADS_PER_BLOCK / lines->size();
        int numBlocks = numPixels / pixelsPerBlock;
        int pixelWorkersPerBlock = pixelsPerBlock * lines->size();
        
        cudaImg = (Pixel*)CudaMalloc(numPixels * sizeof(Pixel));
        CudaMemcpy(cudaImg, image->pixels, numPixels * sizeof(Pixel), cudaMemcpyHostToDevice);

        cudaLines = (Line*)CudaMalloc(lines->size() * sizeof(Line));
        CudaMemcpy(cudaLines, lines->data(), lines->size() * sizeof(Line), cudaMemcpyHostToDevice);

        printf("\nNumber of blocks: %d\n", numBlocks);
        printf("pixelsPerBlock: %d\n", pixelsPerBlock);

        // launch cuda kernal and wait for it to finish
        dim3 grid(numBlocks);
        dim3 block(pixelsPerBlock, lines->size());
        RenderKernel<<<grid, block, pixelWorkersPerBlock * sizeof(int)>>>(cudaImg, image->width, image->height, color, cudaLines, lines->size());
        CudaCheckLaunchErrors();

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