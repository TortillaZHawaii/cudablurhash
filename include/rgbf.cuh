#pragma once

#include <stdint.h>
#include "rgbxy.cuh"

namespace CuBlurHash
{
    struct RGBf
    {
        double r;
        double g;
        double b;
        
        __host__ __device__ RGBf() : r(0), g(0), b(0) {}
        __host__ __device__ RGBf(double r, double g, double b) : r(r), g(g), b(b) {}
        __host__ __device__ RGBf(const RGBf& other) : r(other.r), g(other.g), b(other.b) {}

        __host__ __device__ RGBf operator+ (const RGBf& other) const
        {
            return RGBf(r + other.r, g + other.g, b + other.b);
        }

        __host__ __device__ RGBf operator- (const RGBf& other) const
        {
            return RGBf(r - other.r, g - other.g, b - other.b);
        }

        __host__ __device__ RGBf operator* (const double& other) const
        {
            return RGBf(r * other, g * other, b * other);
        }

        __host__ __device__ RGBf operator/ (const double& other) const
        {
            return RGBf(r / other, g / other, b / other);
        }
    };
}
