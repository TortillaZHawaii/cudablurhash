#pragma once

#include <stdint.h>
#include "rgbxy.cuh"

namespace CuBlurHash
{
    struct RGBf
    {
        float r;
        float g;
        float b;
        
        __host__ __device__ RGBf() : r(0), g(0), b(0) {}
        __host__ __device__ RGBf(float r, float g, float b) : r(r), g(g), b(b) {}
        __host__ __device__ RGBf(const RGBf& other) : r(other.r), g(other.g), b(other.b) {}

        __host__ __device__ RGBf operator+ (const RGBf& other) const
        {
            return RGBf(r + other.r, g + other.g, b + other.b);
        }

        __host__ __device__ RGBf operator- (const RGBf& other) const
        {
            return RGBf(r - other.r, g - other.g, b - other.b);
        }

        __host__ __device__ RGBf operator* (const float& other) const
        {
            return RGBf(r * other, g * other, b * other);
        }

        __host__ __device__ RGBf operator/ (const float& other) const
        {
            return RGBf(r / other, g / other, b / other);
        }
    };
}
