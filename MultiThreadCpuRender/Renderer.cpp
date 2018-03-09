#include "stdafx.h"
#include "Renderer.h"

void Renderer::NaiveLineDrawer(Image* image, Pixel color, std::vector<Line>* lines)
{
	for (int i = 0; i < lines->size(); i++)
	{
		Line line = lines->at(i);
		NaiveDrawLine(image, line, color);
	}
}

void Renderer::NaiveDrawLine(Image* image, Line line, Pixel color)
{
	int tmpdx = line.end.x - line.start.x;
	int tmpdy = line.end.y - line.start.y;

	if (abs((long)tmpdx) < abs((long)tmpdy))
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
		for (unsigned int y = start.y; y < end.y; y++)
		{
			//DrawPixelsAt(image, color, x, y);
			DrawPixelsAt(image, line, color, ceil(x), y);
			DrawPixelsAt(image, line, color, floor(x), y);
			x += dx / (float)dy;
		}
	}
	else
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

		float y = (float)start.y;
		for (unsigned int x = start.x; x < end.x; x++)
		{
			//DrawPixelsAt(image, color, x, y);
			DrawPixelsAt(image, line, color, x, ceil(y));
			DrawPixelsAt(image, line, color, x, floor(y));
			y += dy / (float)dx;
		}
	}
}

void Renderer::DrawPixelsAt(Image* image, Line line, Pixel color, int x, int y)
{
	int opacity = Image::RenderPixel(image, line, x, y);
	if (opacity > 0)
	{
		color.a = opacity;
	}
	auto originalColor = image->GetPixel(x, y);
	*originalColor = Image::BlendPixels(*originalColor, color);
}

void Renderer::DrawPixelsAt(Image* image, Pixel color, float x, float y)
{
	int lastY = (int)ceil(y);
	int firstY = (int)floor(y);
	float percent = y - firstY;

	int intX = (int)floor(x);

	if (percent < 1.0)
	{
		auto lowerPixel = image->GetPixel(intX, firstY);
		color.a = (int)((1.0f - percent) * 255.0f);
		*lowerPixel = Image::BlendPixels(*lowerPixel, color);
	}
	else if (percent > 0)
	{
		auto upperPixel = image->GetPixel(intX, lastY);
		color.a = (int)(percent * 255.0f);
		*upperPixel = Image::BlendPixels(*upperPixel, color);
	}
}
