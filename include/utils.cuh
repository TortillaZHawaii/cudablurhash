#pragma once

#include <math.h>
#include <stdint.h>
#include "rgbf.cuh"
#include "rgbi.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

namespace CuBlurHash
{

    __device__ __host__ uint8_t clamp_to_uint8(int value);

    __device__ __host__ uint8_t linear_to_sRGB(double value);
    __device__ __host__ RGBi linear_to_sRGB(RGBf value);

    __device__ __host__ double sRGB_to_linear(uint8_t value);
    __device__ __host__ RGBf sRGB_to_linear(RGBi value);
}
