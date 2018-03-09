#pragma once
#include "Image.h"

namespace OpenMpRender
{
	void RenderImage(Image* image, Pixel color, std::vector<Line>* lines);
	void RenderImage2(Image * image, Pixel color, std::vector<Line>* lines);
	void DrawLineXCentric(Image * image, Line line, Pixel color);
	void DrawLineYCentric(Image * image, Line line, Pixel color);
	void DrawPixelsAt(Image * image, Line line, Pixel color, int x, int y);
};

