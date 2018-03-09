#include "stdafx.h"
#include "OpenMpRender.h"

#include <omp.h>
#include <iostream>


void OpenMpRender::RenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
#pragma omp parallel for
	for (int x = 0; x < image->width; x++)
	{
		for (int y = 0; y < image->height; y++)
		{
			Image::RenderPixel(image, color, lines, x, y);
		}
	}
}

void OpenMpRender::RenderImageSingleThread(Image*image, Pixel color, std::vector<Line>* lines)
{
	for (int x = 0; x < image->width; x++)
	{
		for (int y = 0; y < image->height; y++)
		{
			Image::RenderPixel(image, color, lines, x, y);
		}
	}
}

