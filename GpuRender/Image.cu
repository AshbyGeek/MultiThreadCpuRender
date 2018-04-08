#include "Image.cuh"
#include <cstdlib>

Image::Image(int width, int height)
{
    this->width = width;
    this->height = height;
    pixels = (Pixel*)calloc(width*height, sizeof(Pixel));
}

Image::~Image()
{
    free(pixels);
}

void Image::FillColor(Pixel color)
{
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            *(GetPixel(i, j)) = color;
        }
    }
}