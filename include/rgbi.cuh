#pragma once

#include <stdint.h>
#include "rgbxy.cuh"

namespace CuBlurHash
{
    struct RGBi
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        
        __host__ __device__ RGBi() : r(0), g(0), b(0) {}
        __host__ __device__ RGBi(uint8_t r, uint8_t g, uint8_t b) : r(r), g(g), b(b) {}

        __device__ __host__ int to_int() const
        {
            return (r << 16) | (g << 8) | b;
        }        
    };
}
