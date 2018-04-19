#include "LibPngWrapper.h"
#include "Image.cuh"

#include <iostream>
#include <chrono>
#include <cuda.h>

using namespace std;
using namespace std::chrono;

extern void CudaRenderImage(Image* image, Pixel color, std::vector<Line>* lines);

void main()
{
    const int WIDTH = 1920 * 2;
    const int HEIGHT = 1080 * 2;
    const int PADDING = 2;

    const int numRuns = 34;
    
    std::vector<Image*> images(numRuns);
    for (int i = 0; i < numRuns; i++)
    {
        images[i] = new Image(WIDTH, HEIGHT);
        Pixel color;
        color.Init(100, 100, 100);
        images[i]->FillColor(color);
    }

    std::vector<Line> lines;
    Line line;

    int centerX = WIDTH / 2;
    int centerY = HEIGHT / 2;
    int length = centerX - PADDING;
    if (centerY < centerX)
    {
        length = centerY - PADDING;
    }

    for (int angle = 0; angle < 360; angle++)
    {
        line.start.x = centerX;
        line.start.y = centerY;

        line.end.x = round(length * sin(angle * 3.1415f / 180)) + line.start.x;
        line.end.y = round(length * cos(angle * 3.1415f / 180)) + line.start.y;
        lines.push_back(line);
    }

    //// diagonal top left to bottom right
    //line.start.x = PADDING;
    //line.start.y = PADDING;
    //line.end.x = WIDTH - PADDING;
    //line.end.y = HEIGHT - PADDING;
    //lines.push_back(line);

    //// diagonal top right to bottom left
    //line.start.x = WIDTH;
    //line.start.y = 0;
    //line.end.x = 0;
    //line.end.y = HEIGHT;
    //lines.push_back(line);

    //// diagonal mid top left to mid bottom right (1/3 slope)
    //line.start.x = 0;
    //line.start.y = (HEIGHT / 3);
    //line.end.x = WIDTH;
    //line.end.y = (HEIGHT * 2 / 3);
    //lines.push_back(line);

    //// vertical line
    //line.start.x = WIDTH / 2;
    //line.end.x = WIDTH / 2;
    //line.start.y = PADDING;
    //line.end.y = HEIGHT - PADDING;
    //lines.push_back(line);

    Pixel color;
    color.R = 0;
    color.G = 0;
    color.B = 0;
    
    auto t1 = high_resolution_clock::now();

    for (int i = 0; i < images.size(); i++)
    {
        std::cout << "Drawing image... ";

        CudaRenderImage(images[i], color, &lines);

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
    for (int i = 0; i < images.size(); i++)
    {
        images[i]->~Image();
        free(images[i]);
    }

    std::cin.get();
}