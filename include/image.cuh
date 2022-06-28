#pragma once

#include <vector>
#include <thrust/host_vector.h>
#include "rgbxy.cuh"
#include <png.h>

namespace CuBlurHash
{
    class Image
    {
        thrust::host_vector<RGBXY> pixels;
        int width, height, channels;

        public:
        Image(std::string const& filename);

        thrust::host_vector<RGBXY> get_pixels() const;
        int get_width() const;
        int get_height() const; 
    };
}