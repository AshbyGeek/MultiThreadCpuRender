#include "stdafx.h"
#include "PThreadsRenderer.h"
#include <Windows.h>
#include <iostream>

PThreadsRenderer::PThreadsRenderer(int numThreads)
{
	std::cout << "Spawning threads\n";

	threads.resize(numThreads);
	for (int i = 0; i < numThreads; i++)
	{
		ThreadData data;
		data.jobsWaitCond = &jobsWaitCond;
		data.jobsWaitCondMutex = &jobsMutex;
		data.stop = &stopThreads;
		data.stopped = false;
		data.threadNum = i;
		data.numThreads = numThreads;

		threads[i].data = data;
		pthread_create(&threads[i].id, NULL, ThreadPoolRun, &threads[i].data);
	}

	std::cout << "All threads spawned\n";
}

PThreadsRenderer::~PThreadsRenderer()
{
	std::cout << "Closing thread pool\n";

	pthread_mutex_lock(&jobsMutex);
	stopThreads = true;
	pthread_mutex_unlock(&jobsMutex);

	while (!AreAllStopped())
	{
		pthread_mutex_lock(&jobsMutex);
		pthread_cond_signal(&jobsWaitCond);
		pthread_mutex_unlock(&jobsMutex);
	}
	std::cout << "Thread pool closed\n";
}

bool PThreadsRenderer::AreAllStopped()
{
	for (int i = 0; i < this->threads.size(); i++)
	{
		if (!threads[i].data.stopped)
		{
			return false;
		}
	}
	return true;
}

void* PThreadsRenderer::ThreadPoolRun(void* voidData)
{
	ThreadData* data = (ThreadData*)voidData;

	while (!*(data->stop))
	{
		pthread_mutex_lock(data->jobsWaitCondMutex);
		pthread_cond_wait(data->jobsWaitCond, data->jobsWaitCondMutex);
		pthread_mutex_unlock(data->jobsWaitCondMutex);


		int x = data->threadNum;
		int y = 0;
		while (!*(data->stop) && (y * data->image->width + x) < (data->image->width * data->image->height))
		{
			while (x > data->image->width)
			{
				x -= data->image->width;
				y += 1;
			}
			RenderPixel(data, x, y);
			x += data->numThreads;
		}

		data->stopped = true;
	}
	pthread_exit(nullptr);
	return 0;
}

void PThreadsRenderer::PThreadsRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
	std::cout << "Starting image job... ";

	this->stopThreads = false;
	for(int i = 0; i < threads.size(); i++)
	{
		threads[i].data.color = color;
		threads[i].data.image = image;
		threads[i].data.lines = lines;
		threads[i].data.stopped = false;
	}

	pthread_mutex_lock(&jobsMutex);
	pthread_cond_broadcast(&jobsWaitCond);
	pthread_mutex_unlock(&jobsMutex);

	bool done = false;
	while (!done)
	{
		done = true;
		for (int i = 0; i < threads.size(); i++)
		{
			done &= threads[i].data.stopped;
		}
		Sleep(20);
	}

	std::cout << "done!\n";
}

void PThreadsRenderer::RenderPixel(ThreadData* data, int x, int y)
{
	Pixel colorAtPixel = *data->image->GetPixel(x, y);

 	for (int i = 0; i < data->lines->size(); i++)
	{
		Line line = data->lines->at(i);

		int dx = line.end.x - line.start.x;
		int dy = line.end.y - line.start.y;
					 
		int px = x - line.start.x;
		int py = y - line.start.y;
		
		float y1 = dy / (float)dx * (float)px;
		float dify = abs(y1 - py);
		if (dify != dify) //NaN - weird way to check
		{
			dify = 0;
		}
		if (dify >= 1 || abs(px) < 0 || abs(px) > abs(dx))
		{
			continue;
		}

		data->color.a = (int)round(255 * (1-dify));
		colorAtPixel = BlendPixels(colorAtPixel, data->color);
	}

	auto pixel = data->image->GetPixel(x, y);
	*(pixel) = colorAtPixel;
}

Pixel PThreadsRenderer::BlendPixels(Pixel p1, Pixel p2)
{
	Pixel newPixel;
	newPixel.r = (p2.r * p2.a + p1.r * (255 - p2.a)) / 255;
	newPixel.g = (p2.g * p2.a + p1.g * (255 - p2.a)) / 255;
	newPixel.b = (p2.b * p2.a + p1.b * (255 - p2.a)) / 255;
	newPixel.a = p1.a + p2.a*(255 - p1.a)/255;
	return newPixel;
}