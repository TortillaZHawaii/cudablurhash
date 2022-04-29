#pragma once

#include <stdint.h>
#include "rgbi.cuh"

namespace CuBlurHash
{
    struct RGBXY 
    {
        RGBi rgb;
        uint8_t x;
        uint8_t y;

        RGBXY() {}
        RGBXY(RGBi rgb, uint8_t x, uint8_t y) : rgb(rgb), x(x), y(y) {}
    };
}
