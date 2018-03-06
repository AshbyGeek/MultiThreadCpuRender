#include "stdafx.h"
#include "Image.h"
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
