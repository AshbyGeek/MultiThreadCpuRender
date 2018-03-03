// MultiThreadCpuRender.cpp : Defines the entry point for the console application.
//
#include <cstdlib>
#include "stdafx.h"

int main()
{
    return 0;
}


struct Pixel{
    unsigned short r;
    unsigned short g;
    unsigned short b;
    unsigned short a;
}

struct Point{
    unsigned int x;
    unsigned int y;
}

struct Line{
    Point start;
    Point end;
}

Pixel** CreateMemoryStructure(int resX, int resY){
    return (Pixel**) calloc(resX * resY * sizeof(Pixel));
}

void NaiveLineDrawer(Pixel** image, Line* lines, int numLines){
    for (int i = 0; i < numLines; i++){
        auto line = lines[i];
        NaiveDrawLine(line);
    }
}

void NaiveDrawLine(Line line){
    
}
