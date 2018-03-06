// MultiThreadCpuRender.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <list>

#include "Renderer.h"
#include "LibPngWrapper.h"

int main()
{
	const int WIDTH = 1920 * 2;
	const int HEIGHT = 1080 * 2;
	const int PADDING = 2;
	const int NUMLINES = 4;

	auto renderer = Renderer();
	Image image(WIDTH, HEIGHT);

	auto lines = (Line*)malloc(NUMLINES * sizeof(Line));
	Line* line;

	line = &lines[0];
	line->start.x = PADDING;
	line->start.y = PADDING;
	line->end.x = WIDTH - PADDING;
	line->end.y = HEIGHT - PADDING;

	line = &lines[1];
	line->start.x = WIDTH - PADDING;
	line->start.y = PADDING;
	line->end.x = PADDING;
	line->end.y = HEIGHT - PADDING;

	line = &lines[2];
	line->start.x = PADDING;
	line->start.y = (HEIGHT / 3);
	line->end.x = WIDTH - PADDING;
	line->end.y = (HEIGHT*2/3);

	line = &lines[3];
	line->start.x = WIDTH / 2;
	line->end.x = WIDTH / 2;
	line->start.y = PADDING;
	line->end.y = HEIGHT - PADDING;

	Pixel color;
	color.r = 0;
	color.g = 0;
	color.b = 0;

	renderer.NaiveLineDrawer(&image, lines, NUMLINES, &color);

	LibPngWrapper pngWrapper("test.png");
	pngWrapper.WriteImage(&image);

	free(lines);

	//std::cin.get();
}
