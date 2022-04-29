#pragma once

#include <vector>
#include <thrust/host_vector.h>
#include "rgbxy.cuh"

namespace CuBlurHash
{
    class Image
    {
        unsigned char* data;
        int width, height, channels;

        public:
        Image(std::string const& filename);
        ~Image();

        thrust::host_vector<RGBXY> get_pixels() const;
        int get_width() const;
        int get_height() const; 
    };
}