#pragma once
#include "Image.h"
#include <vector>


#define HAVE_STRUCT_TIMESPEC
#include "pthread.h"


const int MAX_THREADS = 16;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
int done = 0;
int started = 0;

struct ThreadData
{
	std::vector<Line>* lines;
	Image* image;
	Pixel color;
	int pixelX;
	int pixelY;
};


struct ThreadInfo
{
	pthread_t id;
	ThreadData data;
};

Pixel BlendPixels(Pixel p1, Pixel p2)
{
	Pixel newPixel;
	newPixel.r = (p1.r * p2.a + p2.r * (255 - p2.a)) / 255;
	newPixel.g = (p1.g * p2.a + p2.g * (255 - p2.a)) / 255;
	newPixel.b = (p1.b * p2.a + p2.b * (255 - p2.a)) / 255;
	newPixel.a = p1.a + p2.a*(255 - p1.a);
	return newPixel;
}

void* RenderPixel(void* voidData)
{
	ThreadData* data = (ThreadData*)voidData;

	Pixel colorAtPixel = Pixel();

	for (int i = 0; i < data->lines->size(); i++)
	{
		Line* line = &data->lines->at(i);

		int dx = line->end.x - line->start.x;
		int dy = line->end.y - line->start.y;

		int px = data->pixelX - line->start.x;
		int py = data->pixelY - line->start.y;

		float y1 = dx / (float)dy * (float)px;
		if (abs(y1) > 1)
		{
			continue;
		}

		data->color.a = (int)round(255 * abs(y1));
		colorAtPixel = BlendPixels(colorAtPixel, data->color);
	}

	auto pixel = data->image->GetPixel(data->pixelX, data->pixelY);
	*(pixel) = colorAtPixel;

	pthread_mutex_lock(&mutex);
	done++;
	pthread_cond_signal(&cond);
	pthread_mutex_unlock(&mutex);

	return 0;
}

void PThreadsRenderImage(Image* image, Pixel color, std::vector<Line>* lines)
{
	std::vector<ThreadInfo> threads(image->width * image->height);
	for (int i = 0; i < image->width; i++)
	{
		for (int j = 0; j < image->height; j++)
		{
			int index = j * image->width + i;
			threads[index].data.color = color;
			threads[index].data.image = image;
			threads[index].data.lines = lines;
			threads[index].data.pixelX = i;
			threads[index].data.pixelY = j;

			pthread_create(&threads[index].id, NULL, RenderPixel, (void*)&threads[index].data);
			started++;

			// Make sure that we haven't made too many threads yet
			pthread_mutex_lock(&mutex);
			if (started - done > MAX_THREADS)
			{
				pthread_cond_wait(&cond, &mutex);
			}
			pthread_mutex_unlock(&mutex);
		}
	}

	// Wait for all threads to complete
	pthread_mutex_lock(&mutex);
	while (done < started)
	{
		pthread_cond_wait(&cond, &mutex);
	}
	pthread_mutex_unlock(&mutex);
}