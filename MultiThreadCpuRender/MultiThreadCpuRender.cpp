// MultiThreadCpuRender.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdlib.h>

#include "Renderer.h"
#include "LibPngWrapper.h"
#include <iostream>

int main()
{
	const int WIDTH = 50;
	const int HEIGHT = 30;
	const int PADDING = 2;
	const int NUMLINES = 1;

	auto renderer = Renderer();
	Image image(WIDTH, HEIGHT);

	auto lines = (Line*)malloc(NUMLINES * sizeof(Line));

	auto line = &lines[0];
	line->start.x = PADDING;
	line->start.y = PADDING;
	line->end.x = WIDTH - PADDING;
	line->end.y = HEIGHT - PADDING;

	//auto line = &lines[0];
	//line->start.x = WIDTH - PADDING;
	//line->start.y = PADDING;
	//line->end.x = PADDING;
	//line->end.y = HEIGHT - PADDING;

	//line = &lines[2];
	//line->start.x = PADDING;
	//line->start.y = (HEIGHT / 3);
	//line->end.x = WIDTH - PADDING;
	//line->end.y = (HEIGHT*2/3);

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
