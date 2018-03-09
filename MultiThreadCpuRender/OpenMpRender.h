#pragma once
#include "Image.h"

namespace OpenMpRender
{
	void RenderImage(Image* image, Pixel color, std::vector<Line>* lines);

	void RenderImageSingleThread(Image*image, Pixel color, std::vector<Line>* lines);
};

