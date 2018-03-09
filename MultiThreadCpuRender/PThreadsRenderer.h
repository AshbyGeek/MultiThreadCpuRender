#pragma once
#include "Image.h"
#include <vector>
#include <queue>


#define HAVE_STRUCT_TIMESPEC
#include "pthread.h"



struct ThreadJob
{
	ThreadJob(int pixelX, int pixelY)
	{
		this->pixelX = pixelX;
		this->pixelY = pixelY;
	};
	ThreadJob() {};

	int pixelX;
	int pixelY;
};

struct ThreadData
{
	pthread_mutex_t* jobsWaitCondMutex;
	pthread_cond_t* jobsWaitCond;
	bool* stop;
	bool stopped;
	int threadNum;
	int numThreads;
	
	std::vector<Line>* lines;
	Image* image;
	Pixel color;
};

class PThreadsRenderer
{
private:
	struct ThreadInfo
	{
		pthread_t id;
		ThreadData data;
	};

	pthread_mutex_t jobsMutex = PTHREAD_MUTEX_INITIALIZER;
	pthread_cond_t jobsWaitCond = PTHREAD_COND_INITIALIZER;
	bool stopThreads = false;
	bool AreAllStopped();

	static void* ThreadPoolRun(void* voidData);

	std::vector<ThreadInfo> threads;

	static void RenderPixel(ThreadData* data, int x, int y);
	static Pixel BlendPixels(Pixel p1, Pixel p2);

public:
	PThreadsRenderer(int numThreads);
	~PThreadsRenderer();

	void PThreadsRenderImage(Image* image, Pixel color, std::vector<Line>* lines);
};