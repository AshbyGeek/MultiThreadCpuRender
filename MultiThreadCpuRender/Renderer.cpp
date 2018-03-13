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
            DrawPixelsAt(image, line, color, ceil(x), y, x - ceil(x));
            DrawPixelsAt(image, line, color, floor(x), y, x - floor(x));
            x += dx / (float)dy;
        }
    }
    else
    {
		// Determine which end of the line is leftmost
		Point start = line.start;
		Point end = line.end;
		if (line.start.x > line.end.x)
		{
			start = line.end;
			end = line.start;
		}

		// Get the numerator and the denominator of the slope
		int dx = end.x - start.x;
		int dy = end.y - start.y;

		for (unsigned int x = start.x; x < end.x; x++)
		{
			float y = start.y + dy / (float)dx * (x - start.x);
			DrawPixelsAt(image, line, color, x, ceil(y), y - ceil(y));
			DrawPixelsAt(image, line, color, x, floor(y), y - floor(y));
		}
    }
}

void Renderer::DrawPixelsAt(Image* image, Line line, Pixel color, int x, int y, float distFromPoint)
{
    // Calculate opacity from relative distance
	color.a = abs(distFromPoint) * 255;
    if (color.a > 0)
    {
		//set the pixel value
        auto originalColor = image->GetPixel(x, y);
        *originalColor = Image::BlendPixels(*originalColor, color);
    }
}