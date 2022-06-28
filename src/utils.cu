#include "../include/utils.cuh"

namespace CuBlurHash
{
    __device__ __host__ uint8_t clamp_to_uint8(int value)
    {
        if (value < 0)
            return 0;
        if (value > 255)
            return 255;
        return (uint8_t)value;
    }


    __device__ __host__ uint8_t linear_to_sRGB(double value)
    {
        double v = fmaxf(0.0, fminf(1.0, value));

        if (v <= 0.0031308)
            return (uint8_t)(255.0 * 12.92 * v + 0.5);
        return (uint8_t)(255.0 * (1.055 * powf(v, 1.0 / 2.4) - 0.055) + 0.5);
    }

    __device__ __host__ RGBi linear_to_sRGB(RGBf value)
    {
        RGBi result;
        result.r = linear_to_sRGB(value.r);
        result.g = linear_to_sRGB(value.g);
        result.b = linear_to_sRGB(value.b);
        return result;
    }


    __device__ __host__ double sRGB_to_linear(uint8_t value)
    {
        double v = (double)value / 255.0;

        if (v <= 0.04045)
            return v / 12.92;
        return powf((v + 0.055) / 1.055, 2.4);
    }

    __device__ __host__ RGBf sRGB_to_linear(RGBi value)
    {
        RGBf result;
        result.r = sRGB_to_linear(value.r);
        result.g = sRGB_to_linear(value.g);
        result.b = sRGB_to_linear(value.b);
        return result;
    }
}
