// MultiThreadCpuRender.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <list>
#include <chrono>
#include <omp.h>

#include "Renderer.h"
#include "PThreadsRenderer.h"
#include "LibPngWrapper.h"
#include "OpenMpRender.h"

using namespace std;
using namespace std::chrono;

enum method
{
	omp,
	omp2,
	pthread,
	naieve
};

int main()
{
	const int WIDTH = 1920 * 2;
	const int HEIGHT = 1080 * 2;
	const int PADDING = 2;

	const int numThreads = 8;
	const int numRuns = 34;

	const method method = method::omp2;

	std::vector<Image*> images(numRuns);
	for (int i = 0; i < numRuns; i++)
	{
		images[i] = new Image(WIDTH, HEIGHT);
		images[i]->FillColor(Pixel(100, 100, 100));
	}

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

	PThreadsRenderer* renderer = nullptr;
	if (method == method::pthread)
	{
		renderer = new PThreadsRenderer(numThreads);
		cout << "Number of threads: " << numThreads << "\n";
	}
	else if (method == method::omp || method == method::omp2)
	{
		omp_set_num_threads(numThreads);
		int numOmpThreads = 1;

		#pragma omp parallel
		{
			numOmpThreads = omp_get_num_threads();
		}

		cout << "Number of threads: " << numOmpThreads << "\n";
	}

	auto t1 = high_resolution_clock::now();

	for (int i = 0; i < images.size(); i++)
	{
		std::cout << "Drawing image... ";

		if (method == method::pthread)
		{
			renderer->PThreadsRenderImage(images[i], color, &lines);
		}
		else if (method == method::omp)
		{
			OpenMpRender::RenderImage(images[i], color, &lines);
		}
		else if (method == method::omp2)
		{
			OpenMpRender::RenderImage2(images[i], color, &lines);
		}
		else if (method == method::naieve)
		{
			Renderer::NaiveLineDrawer(images[i], color, &lines);
		}

		std::cout << "Done!\n";
	}

	auto t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(t2 - t1).count();
	
	cout << "Avg runtime (" << numRuns << " runs): ";
	printf("%6.2lf", (duration / 1000 / (float)numRuns)); 
	cout << " milliseconds\n";

	LibPngWrapper pngWrapper("test.png");
	pngWrapper.WriteImage(images[0]);

	std::cout << "Wrapping up\n";
	pngWrapper.~LibPngWrapper();
	if (method == method::pthread)
	{
		renderer->~PThreadsRenderer();
		renderer = nullptr;
	}
	for (int i = 0; i < images.size(); i++)
	{
        images[i]->~Image();
		free(images[i]);
	}

	std::cin.get();
}
