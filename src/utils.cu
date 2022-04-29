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


    __device__ __host__ uint8_t linear_to_sRGB(float value)
    {
        float v = fmaxf(0.0f, fminf(1.0f, value));

        if (v <= 0.0031308f)
            return (uint8_t)(255.0f * 12.92f * v + 0.5f);
        return (uint8_t)(255.0f * (1.055f * powf(v, 1.0f / 2.4f) - 0.055f) + 0.5f);
    }

    __device__ __host__ RGBi linear_to_sRGB(RGBf value)
    {
        RGBi result;
        result.r = linear_to_sRGB(value.r);
        result.g = linear_to_sRGB(value.g);
        result.b = linear_to_sRGB(value.b);
        return result;
    }


    __device__ __host__ float sRGB_to_linear(uint8_t value)
    {
        float v = (float)value / 255.0f;

        if (v <= 0.04045f)
            return v / 12.92f;
        return powf((v + 0.055f) / 1.055f, 2.4f);
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
