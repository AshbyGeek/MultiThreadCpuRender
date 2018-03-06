#include "stdafx.h"
#include "Renderer.h"

#include <cstdlib>


Renderer::Renderer()
{
	this->color.r = 255;
	this->color.g = 255;
	this->color.b = 255;
	this->color.a = 255;
}

Renderer::Renderer(Pixel color)
{
	this->color.r = color.r;
	this->color.g = color.g;
	this->color.b = color.b;
	this->color.a = color.a;
}

Renderer::~Renderer()
{
}

void Renderer::NaiveLineDrawer(Image* image, Line* lines, int numLines, Pixel* color)
{
	for (int i = 0; i < numLines; i++)
	{
		auto line = &lines[i];
		NaiveDrawLine(image, line, color);
	}
}

void Renderer::NaiveDrawLine(Image* image, Line* line, Pixel* color)
{
	int tmpdx = line->end.x - line->start.x;
	int tmpdy = line->end.y - line->start.y;

	if (abs((long)tmpdx) < abs((long)tmpdy))
	{
		Point* start = &line->start;
		Point* end = &line->end;
		if (line->start.y > line->end.y)
		{
			start = &line->end;
			end = &line->start;
		}

		int dx = end->x - start->x;
		int dy = end->y - start->y;

		float x = (float)start->x;
		for (unsigned int y = start->y; y < end->y; y++)
		{
			DrawPixelsAt(image, color, (float)start->x, y);
			x += dx / (float)dy;
		}
	}
	else
	{
		Point* start = &line->start;
		Point* end = &line->end;
		if (line->start.x > line->end.x)
		{
			start = &line->end;
			end = &line->start;
		}

		int dx = end->x - start->x;
		int dy = end->y - start->y;

		float y = (float)start->y;
		for (unsigned int x = start->x; x < end->x; x++)
		{
			DrawPixelsAt(image, color, (float)x, y);
			y += dy / (float)dx;
		}
	}
}

void Renderer::DrawPixelsAt(Image* image, Pixel* color, float x, float y)
{
	int lastY = (int)ceil(y);
	int firstY = (int)floor(y);
	float percent = y - firstY;

	int intX = (int)floor(x);

	if (percent < 1.0)
	{
		auto lowerPixel = image->GetPixel(intX, firstY);
		lowerPixel->r = color->r;
		lowerPixel->g = color->g;
		lowerPixel->b = color->b;
		lowerPixel->a = (int)((1.0f - percent) * 255.0f);
	}

	if (percent > 0)
	{
		auto upperPixel = image->GetPixel(intX, lastY);
		upperPixel->r = color->r;
		upperPixel->g = color->g;
		upperPixel->b = color->b;
		upperPixel->a = (int)(percent * 255.0f);
	}
}