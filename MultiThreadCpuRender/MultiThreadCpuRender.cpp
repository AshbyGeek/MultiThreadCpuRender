// MultiThreadCpuRender.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <list>
#include <chrono>

#include "Renderer.h"
#include "PThreadsRenderer.h"
#include "LibPngWrapper.h"

using namespace std;
using namespace std::chrono;

int main()
{
	const int WIDTH = 50;
	const int HEIGHT = 30;
	const int PADDING = 2;

	auto renderer = Renderer();
	Image image(WIDTH, HEIGHT);
	Image image2(WIDTH, HEIGHT);
	Image image3(WIDTH, HEIGHT);
	Image image4(WIDTH, HEIGHT);

	std::vector<Line> lines;

	Line line;
	line.start.x = PADDING;
	line.start.y = PADDING;
	line.end.x = WIDTH - PADDING;
	line.end.y = HEIGHT - PADDING;
	lines.push_back(line);

	line.start.x = WIDTH - PADDING;
	line.start.y = PADDING;
	line.end.x = PADDING;
	line.end.y = HEIGHT - PADDING;
	lines.push_back(line);
		
	line.start.x = PADDING;
	line.start.y = (HEIGHT / 3);
	line.end.x = WIDTH - PADDING;
	line.end.y = (HEIGHT*2/3);
	lines.push_back(line);
		
	line.start.x = WIDTH / 2;
	line.end.x = WIDTH / 2;
	line.start.y = PADDING;
	line.end.y = HEIGHT - PADDING;
	lines.push_back(line);

	Pixel color;
	color.r = 0;
	color.g = 0;
	color.b = 0;

	auto t1 = high_resolution_clock::now();

	PThreadsRenderImage(&image, color, &lines);
	PThreadsRenderImage(&image2, color, &lines);
	PThreadsRenderImage(&image3, color, &lines);
    PThreadsRenderImage(&image4, color, &lines);

	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1).count();
    cout << "Avg runtime (4 runs): ";
    printf("%6.2lf", (duration / 1000 / 4.0f));
    cout << "\n";

	LibPngWrapper pngWrapper("test.png");
	pngWrapper.WriteImage(&image);

	pngWrapper.~LibPngWrapper();
	renderer.~Renderer();
    image.~Image();
    image2.~Image();
    image3.~Image();
    image4.~Image();

	std::cin.get();
}
