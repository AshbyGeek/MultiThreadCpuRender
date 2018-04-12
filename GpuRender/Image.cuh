#pragma once

#include <vector>
#include "cuda_runtime.h"

struct Pixel
{
    __device__
    __host__
    void Init(unsigned char r, unsigned char g, unsigned char b, unsigned char a = 255)
    {
        this->R = r;
        this->G = g;
        this->B = b;
        this->A = a;
    }
    
    unsigned char R = 255;
    unsigned char G = 255;
    unsigned char B = 255;
    unsigned char A = 255;
};

struct Point
{
    __device__
    __host__
    Point(float x, float y)
    {
        this->x = x;
        this->y = y;
    }
    
    __device__
    __host__
    Point() {};

    float x;
    float y;

    __device__
    __host__
    Point operator+(const Point& b)
    {
        Point pt;
        pt.x = x + b.x;
        pt.y = y + b.y;
        return pt;
    }

    __device__
    __host__
    Point operator-(const Point& b)
    {
        Point pt;
        pt.x = x - b.x;
        pt.y = y - b.y;
        return pt;
    }
    
    __device__
    __host__
    Point operator*(const float& b)
    {
        Point pt;
        pt.x = x * b;
        pt.y = y * b;
        return pt;
    }
    
    __device__
    __host__
    static float CrossProduct(Point a, Point b)
    {
        return a.x * b.y - a.y * b.x;
    }
    
    __device__
    __host__
    static float DotProduct(Point a, Point b)
    {
        return a.x * b.x + a.y * b.y;
    }
    
    __device__
    __host__
    static Point Vector(Point a, Point b)
    {
        Point vect;
        vect.x = b.x - a.x;
        vect.y = b.y - a.y;
        return vect;
    }

    __device__
    __host__
    static float LengthSquared(Point vector)
    {
        return vector.x * vector.x + vector.y * vector.y;
    }
    
    __device__
    __host__
    static float Length(Point vector)
    {
        return sqrtf(vector.x * vector.x + vector.y * vector.y);
    }
};

struct Line
{
    __device__
    __host__
    Line() {}

    __device__
    __host__
    Line(Point start, Point end)
    {
        this->start = start;
        this->end = end;
    }

    Point start;
    Point end;
    
    __device__
    __host__
    Point Vector()
    {
        return Point::Vector(start, end);
    }
    
    __device__
    __host__
    float LengthSquared()
    {
        return Point::LengthSquared(Vector());
    }
    
    __device__
    __host__
    float Length()
    {
        return Point::Length(Vector());
    }
    
    __device__
    __host__
    float DistSquaredTo(Point p)
    {
        Point a = start;
        Point b = end;

        // Lots of vector math
        Point n = b - a;
        Point pa = a - p;

        float c = Point::DotProduct(n, pa);
        if (c > 0.0f)
            return Point::LengthSquared(pa);

        Point bp = p - b;
        if (Point::DotProduct(n, bp) > 0.0f)
            return Point::LengthSquared(bp);

        Point e = pa - n * (c / Point::LengthSquared(n));
        return Point::LengthSquared(e);
    }
};


class Image
{
public:
    int width;
    int height;
    Pixel* pixels;
    
    inline Pixel* GetPixel(int x, int y)
    {
        return &pixels[y * width + x];
    }
    
    Image(int width, int height);
    ~Image();
    
    void FillColor(Pixel color);
};

