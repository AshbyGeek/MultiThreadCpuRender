#pragma once
#include "cuda_runtime.h"
#include "Image.cuh"

__device__ 
Pixel BlendPixels(Pixel p1, Pixel p2);

__device__ 
Pixel RenderPixel(int x, int y, Pixel color, Line * lineArray, int numLines);

__global__ 
void RenderKernel(Pixel * pixels, int imgWidth, int imgHeight, Pixel color, Line * lineArray, int numLines);

void CudaRenderImage(Image * image, Pixel color, std::vector<Line>* lines);
