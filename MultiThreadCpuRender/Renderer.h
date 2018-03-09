#pragma once
#include <vector>

#include "Image.h"
#include "OpenMpRender.h"

namespace Renderer
{
	void NaiveLineDrawer(Image* image, Pixel color, std::vector<Line>* lines);

	void NaiveDrawLine(Image* image, Line line, Pixel color);

	void DrawPixelsAt(Image* image, Line line, Pixel color, int x, int y);

	void DrawPixelsAt(Image* image, Pixel color, float x, float y);
}
