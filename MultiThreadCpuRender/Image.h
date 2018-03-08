#pragma once

struct Pixel
{
	Pixel(char r, char g, char b, char a = 255)
	{
		this->r = r;
		this->g = g;
		this->b = b;
		this->a = a;
	}

	Pixel() {};

	unsigned char r = 255;
	unsigned char g = 255;
	unsigned char b = 255;
	unsigned char a = 255;
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

	void FillColor(Pixel color);
};

