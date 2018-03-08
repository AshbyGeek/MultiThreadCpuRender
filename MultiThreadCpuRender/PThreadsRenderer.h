#pragma once
#include "Image.h"
#include <vector>
#include <queue>


#define HAVE_STRUCT_TIMESPEC
#include "pthread.h"



struct ThreadJob
{
	std::vector<Line>* lines;
	Image* image;
	Pixel color;
	int pixelX;
	int pixelY;
};

struct ThreadData
{
	pthread_mutex_t* jobsWaitCondMutex;
	pthread_cond_t* jobsWaitCond;
	std::queue<ThreadJob>* jobs;
	bool* stop;
	bool stopped;
	int threadNum;
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
	std::queue<ThreadJob> jobs;

	void queue_add(ThreadJob value);
	static ThreadJob queue_get(std::queue<ThreadJob>* jobs, pthread_mutex_t* jobsMutex, pthread_cond_t* jobsCond, bool* cancel);
	void waitQueueEmpty();

	static void RenderPixel(ThreadJob job);
	static Pixel BlendPixels(Pixel p1, Pixel p2);

public:
	PThreadsRenderer(int numThreads);
	~PThreadsRenderer();

	void PThreadsRenderImage(Image* image, Pixel color, std::vector<Line>* lines);
};