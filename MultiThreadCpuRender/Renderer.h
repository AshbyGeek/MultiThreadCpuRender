#pragma once

#include "Image.h"

struct Point
{
	unsigned int x;
	unsigned int y;
};

struct Line
{
	Point start;
	Point end;
};

class Renderer
{
public:
	Renderer();
	Renderer(Pixel color);
	~Renderer();

	Pixel color;

	void NaiveLineDrawer(Image* image, Line* lines, int numLines, Pixel* color);

	void NaiveDrawLine(Image* image, Line* line, Pixel* color);

	void DrawPixelsAt(Image* image, Pixel* color, float x, float y, bool vertical);
};



