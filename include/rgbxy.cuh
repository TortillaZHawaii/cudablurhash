#pragma once

#include <stdint.h>
#include "rgbi.cuh"

namespace CuBlurHash
{
    struct RGBXY 
    {
        RGBi rgb;
        int x;
        int y;

        RGBXY() {}
        RGBXY(RGBi rgb, int x, int y) : rgb(rgb), x(x), y(y) {}
    };
}
