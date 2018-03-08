#include "stdafx.h"
#include "PThreadsRenderer.h"
#include <Windows.h>
#include <iostream>

PThreadsRenderer::PThreadsRenderer(int numThreads)
{
	std::cout << "Spawning threads\n";

	threads.resize(numThreads);
	for (int i = 0; i < threads.size(); i++)
	{
		ThreadData data;
		data.jobs = &jobs;
		data.jobsWaitCond = &jobsWaitCond;
		data.jobsWaitCondMutex = &jobsMutex;
		data.stop = &stopThreads;
		data.stopped = false;
		data.threadNum = i;

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
	jobs.empty();
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
		ThreadJob job = PThreadsRenderer::queue_get(data->jobs, data->jobsWaitCondMutex, data->jobsWaitCond, data->stop);
		if (!*(data->stop))
		{
			RenderPixel(job);
		}
	}
	data->stopped = true;
	pthread_exit(nullptr);
	return 0;
}

void PThreadsRenderer::queue_add(ThreadJob value)
{
	pthread_mutex_lock(&jobsMutex);
	this->jobs.push(value);
	pthread_cond_signal(&jobsWaitCond);
	pthread_mutex_unlock(&jobsMutex);
}


ThreadJob PThreadsRenderer::queue_get(std::queue<ThreadJob>* jobs, pthread_mutex_t* jobsMutex, pthread_cond_t* jobsCond, bool* cancel)
{
	pthread_mutex_lock(jobsMutex);
	while (jobs->empty() && !*(cancel))
	{
		pthread_cond_wait(jobsCond, jobsMutex);
	}

	if (jobs->empty())
	{
		pthread_mutex_unlock(jobsMutex);
		return ThreadJob();
	}
	else
	{
		ThreadJob job = jobs->front();
		jobs->pop();
		pthread_mutex_unlock(jobsMutex);
		return job;
	}
}

void PThreadsRenderer::waitQueueEmpty()
{
	while (!jobs.empty())
	{
		Sleep(10);
	}

	std::cout << "Queue is empty\n";
}


void PThreadsRenderer::PThreadsRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
	int done = 0;
	int started = 0;

	std::cout << "Starting to queue jobs\n";

	std::vector<ThreadInfo> threads(image->width * image->height);
	for (int i = 0; i < image->width; i++)
	{
		for (int j = 0; j < image->height; j++)
		{
			ThreadJob job;
			job.color = color;
			job.image = image;
			job.lines = lines;
			job.pixelX = i;
			job.pixelY = j;

			queue_add(job);
		}
	}

	std::cout << "Done queing jobs\n";

	waitQueueEmpty();
}

void PThreadsRenderer::RenderPixel(ThreadJob job)
{
	Pixel colorAtPixel = *job.image->GetPixel(job.pixelX, job.pixelY);


 	for (int i = 0; i < job.lines->size(); i++)
	{
		Line line = job.lines->at(i);

		int dx = line.end.x - line.start.x;
		int dy = line.end.y - line.start.y;
					 
		int px = job.pixelX - line.start.x;
		int py = job.pixelY - line.start.y;
		
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

		job.color.a = (int)round(255 * (1-dify));
		colorAtPixel = BlendPixels(colorAtPixel, job.color);
	}

	auto pixel = job.image->GetPixel(job.pixelX, job.pixelY);
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