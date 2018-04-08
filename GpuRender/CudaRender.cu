#include "cuda_runtime.h"
#include <vector>

#include "Image.cuh"
#include "CudaWrapper.cuh"
#include "CudaRender.cuh"

const int THREADS_PER_BLOCK = 1024;

using namespace CudaWrapper;

__device__
Pixel BlendPixels(Pixel p1, Pixel p2)
{
    Pixel newPixel;
    newPixel.r = (p1.r * p2.a + p2.r * (255 - p2.a)) / 255;
    newPixel.g = (p1.g * p2.a + p2.g * (255 - p2.a)) / 255;
    newPixel.b = (p1.b * p2.a + p2.b * (255 - p2.a)) / 255;
    newPixel.a = p1.a + p2.a*(255 - p1.a);
    return newPixel;
}

__device__
Pixel RenderPixel(Point pt, Pixel color, Line* lineArray, int numLines)
{
    Pixel colorAtPixel = Pixel();

    for (int i = 0; i < numLines; i++)
    {
        Line line = lineArray[i];

        Point vectPtStart = pt - line.start;
        Point vectEndStart = line.end - line.start;
        float t = Point::DotProduct(vectPtStart, vectEndStart) / Point::DotProduct(vectEndStart, vectEndStart);
        float distSquared = Point::LengthSquared(vectPtStart - vectEndStart * t);

        if (abs(distSquared) <= 1)
        {
            color.a = (int)round(255 * abs(distSquared));
            colorAtPixel = BlendPixels(colorAtPixel, color);
        }
    }

    return colorAtPixel;
}

__global__
void RenderKernel(Pixel* pixels, int imgWidth, int imgHeight, Pixel color, Line* lineArray, int numLines)
{
    int pixelNum = threadIdx.x + blockIdx.x * blockDim.x;
    int x = pixelNum / imgWidth;
    int y = pixelNum % imgWidth;

    if (x > imgWidth || y > imgHeight)
        return;

    Point pt;
    pt.x = x;
    pt.y = y;

    auto tmpColor = RenderPixel(pt, color, lineArray, numLines);
    pixels[pixelNum] = tmpColor;
}

void CudaRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
    Pixel* cudaImg = nullptr;
    Line* cudaLines = nullptr;
    try
    {
        // allocate memory
        int numPixels = image->width * image->height;
        
        cudaImg = (Pixel*)CudaMalloc(numPixels * sizeof(Pixel));
        CudaMemcpy(cudaImg, image->pixels, numPixels * sizeof(Pixel), cudaMemcpyHostToDevice);

        cudaLines = (Line*)CudaMalloc(lines->size() * sizeof(Line));
        CudaMemcpy(cudaLines, lines->data(), lines->size() * sizeof(Line), cudaMemcpyHostToDevice);

        // Figure out how many threads and blocks
        int numBlocks = numPixels / THREADS_PER_BLOCK;
        int numThreads = THREADS_PER_BLOCK;

        // launch cuda kernal and wait for it to finish
        CudaSetDevice(0);
        RenderKernel<<<numBlocks, numThreads>>>(cudaImg, image->width, image->height, color, cudaLines, lines->size());
        CudaCheckLaunchErrors();
        CudaDeviceSynchronize();

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
}