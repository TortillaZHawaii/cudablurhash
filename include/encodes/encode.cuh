#pragma once

#include <string>
#include "../image.cuh"

namespace CuBlurHash
{
    std::string encode_image(
        CuBlurHash::Image const& image,
        int const& x_compound, 
        int const& y_compound
        );
}
