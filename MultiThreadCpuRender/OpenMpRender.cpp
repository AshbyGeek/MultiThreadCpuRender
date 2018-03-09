#include "stdafx.h"
#include "OpenMpRender.h"

#include <omp.h>
#include <iostream>


void OpenMpRender::RenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
	int x,y;

	#pragma omp parallel for shared(image, color, lines) \
							 private(x,y) \
							 schedule(static)
	for (int i = 0; i < image->width * image->height; i++)
	{
		x = i / image->height;
		y = i % image->height;
		Image::RenderPixel(image, color, lines, x, y);
	}
}


void OpenMpRender::RenderImage2(Image* image, Pixel color, std::vector<Line>* lines)
{
	for (int i = 0; i < lines->size(); i++)
	{
		auto line = lines->at(i);

		int tmpdx = line.end.x - line.start.x;
		int tmpdy = line.end.y - line.start.y;

		if (abs((long)tmpdx) < abs((long)tmpdy))
		{
			DrawLineXCentric(image, line, color);
		}
		else
		{
			DrawLineYCentric(image, line, color);
		}
	}
}

void OpenMpRender::DrawLineXCentric(Image* image, Line line, Pixel color)
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

	float x = (float)start.x;

	#pragma omp parallel for
	for (int y = start.y; y < end.y; y++)
	{
		DrawPixelsAt(image, line, color, ceil(x), y);
		DrawPixelsAt(image, line, color, floor(x), y);
		x = start.x + dx / (float)dy * (y - start.y);
	}
}

void OpenMpRender::DrawLineYCentric(Image* image, Line line, Pixel color)
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

	#pragma omp parallel for
	for (int x = start.x; x < end.x; x++)
	{
		float y = start.y + dy / (float)dx * (x - start.x);
		DrawPixelsAt(image, line, color, x, ceil(y));
		DrawPixelsAt(image, line, color, x, floor(y));
	}
}

void OpenMpRender::DrawPixelsAt(Image* image, Line line, Pixel color, int x, int y)
{
	int opacity = Image::RenderPixel(image, line, x, y);
	if (opacity > 0)
	{
		color.a = opacity;
	}
	auto originalColor = image->GetPixel(x, y);
	*originalColor = Image::BlendPixels(*originalColor, color);
}
