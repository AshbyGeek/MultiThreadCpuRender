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
	const int WIDTH = 1920 * 2;
	const int HEIGHT = 1080 * 2;
	const int PADDING = 2;

	Image image(WIDTH, HEIGHT);
	image.FillColor(Pixel(100, 100, 100));

	Image image2(WIDTH, HEIGHT);
	Image image3(WIDTH, HEIGHT);
	Image image4(WIDTH, HEIGHT);

	std::vector<Line> lines;

	Line line;

	// diagonal top left to bottom right
	line.start.x = PADDING;
	line.start.y = PADDING;
	line.end.x = WIDTH - PADDING;
	line.end.y = HEIGHT - PADDING;
	lines.push_back(line);

	// diagonal top right to bottom left
	line.start.x = WIDTH;
	line.start.y = 0;
	line.end.x = 0;
	line.end.y = HEIGHT;
	lines.push_back(line);
	
	// diagonal mid top left to mid bottom right (1/3 slope)
	line.start.x = 0;
	line.start.y = (HEIGHT / 3);
	line.end.x = WIDTH;
	line.end.y = (HEIGHT*2/3);
	lines.push_back(line);

	// vertical line
	line.start.x = WIDTH / 2;
	line.end.x = WIDTH / 2;
	line.start.y = PADDING;
	line.end.y = HEIGHT - PADDING;
	lines.push_back(line);

	Pixel color;
	color.r = 0;
	color.g = 0;
	color.b = 0;

	PThreadsRenderer renderer(1);

	auto t1 = high_resolution_clock::now();

	renderer.PThreadsRenderImage(&image, color, &lines);
	renderer.PThreadsRenderImage(&image2, color, &lines);
	renderer.PThreadsRenderImage(&image3, color, &lines);

	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	cout << "Avg runtime (4 runs): " << duration / 4.0 << "\n";

	LibPngWrapper pngWrapper("test.png");
	pngWrapper.WriteImage(&image);


	std::cout << "Wrapping up\n";
	pngWrapper.~LibPngWrapper();
	renderer.~PThreadsRenderer();

	std::cin.get();
}
