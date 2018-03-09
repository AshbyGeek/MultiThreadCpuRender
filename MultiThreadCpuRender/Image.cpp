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


Pixel Image::BlendPixels(Pixel p1, Pixel p2)
{
	Pixel newPixel;
	newPixel.r = (p2.r * p2.a + p1.r * (255 - p2.a)) / 255;
	newPixel.g = (p2.g * p2.a + p1.g * (255 - p2.a)) / 255;
	newPixel.b = (p2.b * p2.a + p1.b * (255 - p2.a)) / 255;
	newPixel.a = p1.a + p2.a*(255 - p1.a) / 255;
	return newPixel;
}


void Image::RenderPixel(Image* image, Pixel color, std::vector<Line>* lines, int x, int y)
{
	Pixel colorAtPixel = *image->GetPixel(x, y);

	for (int i = 0; i < lines->size(); i++)
	{
		Line line = lines->at(i);
		int opacity = RenderPixel(image, line, x, y);
		if (opacity > 0)
		{
			color.a = opacity;
			colorAtPixel = Image::BlendPixels(colorAtPixel, color);
		}
	}

	auto pixel = image->GetPixel(x, y);
	*(pixel) = colorAtPixel;
}

/// <summary>
/// Returns the opacity value to use at the given pixel
/// </summary>
/// <param name="image"></param>
/// <param name="line"></param>
/// <param name="x"></param>
/// <param name="y"></param>
/// <returns></returns>
int Image::RenderPixel(Image* image, Line line, int x, int y)
{
	int dx = line.end.x - line.start.x;
	int dy = line.end.y - line.start.y;

	int px = x - line.start.x;
	int py = y - line.start.y;

	float y1 = dy / (float)dx * (float)px;
	float dify = abs(y1 - py);
	if (dify != dify) //NaN - weird way to check
	{
		dify = 0;
	}
	if (dify >= 1 || abs(px) < 0 || abs(px) > abs(dx))
	{
		return 0;
	}
	else
	{
		return (int)round(255 * (1 - dify));
	}
}
