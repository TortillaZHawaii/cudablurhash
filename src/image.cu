#include "../include/image.cuh"
extern "C" {
    #define STB_IMAGE_IMPLEMENTATION
    #include "../include/stb/stb_image.h"
    #include "../include/stb/stb_writer.h"
}

namespace CuBlurHash
{
    Image::Image(std::string const& filename)
    {
        data = stbi_load(filename.c_str(), &width, &height, &channels, 3);
    }

    Image::~Image()
    {
        stbi_image_free(data);
    }

    thrust::host_vector<RGBXY> Image::get_pixels() const
    {
        thrust::host_vector<RGBXY> pixels(width * height);
        for (int i = 0; i < width * height; i++)
        {
            pixels[i].x = i % width;
            pixels[i].y = i / width;
            pixels[i].rgb.r = data[i * 3 + 0];
            pixels[i].rgb.g = data[i * 3 + 1];
            pixels[i].rgb.b = data[i * 3 + 2];
        }
        return pixels;
    }

    int Image::get_width() const
    {
        return width;
    }

    int Image::get_height() const
    {
        return height;
    }
}