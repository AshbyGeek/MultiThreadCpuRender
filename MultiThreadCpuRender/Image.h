#pragma once

struct Pixel
{
	unsigned char r;
	unsigned char g;
	unsigned char b;
	unsigned char a;
};

struct Point
{
	unsigned int x;
	unsigned int y;
};

struct Line
{
	Point start;
	Point end;
};


class Image
{
public:
	int width;
	int height;
	Pixel* pixels;

	inline Pixel* GetPixel(int x, int y)
	{
		if (x > width)
			throw "x is outside of image!";
		
		if (y > height)
			throw "y is outside of image!";

		return &pixels[y * width + x];
	}

	Image(int width, int height);
	~Image();
};

