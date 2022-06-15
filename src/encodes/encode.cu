#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <execution>

#include "../../include/encodes/encode.cuh"
#include "../../include/rgbxy.cuh"
#include "../../include/rgbf.cuh"
#include "../../include/utils.cuh"

namespace CuBlurHash
{
    const char* hash_chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz#$%*+,-.:;=?@[]^_{|}~";

    thrust::host_vector<RGBf> get_factors(
        int const& x_components,
        int const& y_components,
        int const& width,
        int const& height,
        thrust::device_vector<RGBXY> const& d_rgb_vector
        );

    std::string encode_factors(
        int const& x_components,
        int const& y_components,
        thrust::host_vector<RGBf> const& factors
        );

    __host__ __device__ inline int encode_dc(RGBf const& rgb);
    __host__ __device__ inline int encode_ac(RGBf const& rgb, float max_value);
    std::string encode_int(
        int value,
        int length
        );

    std::string encode_image(
        CuBlurHash::Image const& image,
        int const& x_components, 
        int const& y_components
        )
    {
        // std::cout << "Getting image info..." << std::endl;

        auto h_rgb_vector = image.get_pixels();
        int width = image.get_width();
        int height = image.get_height();

        // std::cout << "Image size: " << width << "x" << height << std::endl;

        // std::cout << "Copying image to device..." << std::endl;

        thrust::device_vector<RGBXY> d_rgb_vector = h_rgb_vector;

        // std::cout << "Getting factors..." << std::endl;

        auto factors = get_factors(x_components, y_components, width, height, d_rgb_vector);

        // std::cout << "Encoding factors..." << std::endl;

        return encode_factors(x_components, y_components, factors);
    }

    struct basis : public std::unary_function<RGBXY, RGBf>
    {
        int x_components;
        int y_components;
        int width;
        int height;

        basis(int const& x_components, int const& y_components, int const& width, int const& height)
            : x_components(x_components), y_components(y_components), width(width), height(height)
        {
        }

        __host__ __device__ RGBf operator()(RGBXY const& rgbxy) const
        {
            RGBf rgbf = sRGB_to_linear(rgbxy.rgb);

            return rgbf * get_basis(rgbxy.x, rgbxy.y);
        }

        __host__ __device__ inline float get_basis(float const& x, float const& y) const
        {
            return cosf(M_PI * x_components * x / width)
                * cosf(M_PI * y_components * y / height);
        }
    };

    RGBf multiply_basis_function(
        int const& x_component,
        int const& y_component,
        int const& width,
        int const& height,
        thrust::device_vector<RGBXY> const& d_rgb_vector
        )
    {
        RGBf result = thrust::transform_reduce(
            d_rgb_vector.begin(),
            d_rgb_vector.end(),
            basis(x_component, y_component, width, height),
            RGBf(),
            thrust::plus<RGBf>()
            );

        float normalisation = (x_component == 0 && y_component == 0) ? 1.0f : 2.0f;
        float scale = normalisation / (width * height);

        return result * scale;
    }

    thrust::host_vector<RGBf> get_factors(
        int const& x_components,
        int const& y_components,
        int const& width,
        int const& height,
        thrust::device_vector<RGBXY> const& d_rgb_vector
        )
    {
        thrust::host_vector<RGBf> h_basis_vector(x_components * y_components);

        std::vector<int> indexes = std::vector<int>(x_components * y_components);
        std::iota(indexes.begin(), indexes.end(), 0);

        std::for_each(
            indexes.begin(),
            indexes.end(),
            [&](int const& index)
            {
                int x_component = index % x_components;
                int y_component = index / x_components;

                h_basis_vector[index] = multiply_basis_function(x_component, y_component, width, height, d_rgb_vector);
            });

        return h_basis_vector;
    }

    struct max_rgbf_component : public std::unary_function<RGBf, float>
    {
        __host__ __device__ float operator()(RGBf const& rgbf) const
        {
            return fmaxf(fmaxf(fabsf(rgbf.r), fabsf(rgbf.g)), fabsf(rgbf.b));
        }
    };

    std::string encode_factors(
        int const& x_components,
        int const& y_components,
        thrust::host_vector<RGBf> const& factors
        )
    {
        std::string hash = std::string();

        // std::cout << "Encoding size..." << std::endl;
        // encode size
        int size_flag = (x_components - 1) + (y_components - 1) * 9;

        hash += encode_int(size_flag, 1);

        // std::cout << "Calculating max value..." << std::endl;


        float max_value;
        // encode max value
        if(x_components != 1 || y_components != 1)
        {
            float max_component = thrust::transform_reduce(
                factors.begin(),
                factors.end(),
                max_rgbf_component(),
                0.0f,
                thrust::maximum<float>()
            );
            // std::cout << "Max component: " << max_component << std::endl;

            int quantised_max_component = (int)fmaxf(0, fminf(82, floorf(max_component * 166.0f - 0.5f)));
            max_value = (quantised_max_component + 1) / 166.0f;
            // std::cout << "Max value: " << max_value << std::endl;
            // std::cout << "Quantised max value: " << quantised_max_component << std::endl;
            
            // std::cout << "Encoding quantised max value..." << std::endl;
            hash += encode_int(quantised_max_component, 1);
        }
        else 
        {
            max_value = 1.0f;
            // std::cout << "Encoding max value..." << std::endl;
            hash += encode_int(0, 1);
        }

        // std::cout << "Encoding dc part..." << std::endl;
        // encode factors
        // encode dc
        hash += encode_int(encode_dc(factors[0]), 4);

        // std::cout << "Encoding ac part..." << std::endl;
        // encode ac
        for(int i = 1; i < factors.size(); ++i)
        {
            hash += encode_int(encode_ac(factors[i], max_value), 2);
        }

        return hash;
    }

    __host__ __device__ inline int encode_dc(RGBf const& rgb)
    {
        RGBi rgbi = linear_to_sRGB(rgb);
        return rgbi.to_int();
    }

    __host__ __device__ inline float signed_pow(float base, float exponent)
    {
        return copysignf(powf(fabsf(base), exponent), base);
    }

    __host__ __device__ inline int encode_ac_part(float part, float max_value)
    {
        return fmaxf(0,
                fminf(18.0f,
                    floorf(
                        signed_pow(part / max_value, 0.5f)
                        * 9.0f + 9.5f
                    )
                )
            );
    }

    __host__ __device__ inline int encode_ac(RGBf const& rgb, float max_value)
    {
        int quant_r = encode_ac_part(rgb.r, max_value);
        int quant_g = encode_ac_part(rgb.g, max_value);
        int quant_b = encode_ac_part(rgb.b, max_value);

        return quant_r * 19 * 19 + quant_g * 19 + quant_b;
    }

    std::string encode_int(
        int value,
        int length
        )
    {
        std::string encoded = std::string();
        int divisor = 1;

        for(int i = 0; i < length - 1; ++i)
            divisor *= 83;

        for(int i = 0; i < length; ++i)
        {
            int digit = (value / divisor) % 83;
            divisor /= 83;
            encoded += hash_chars[digit];
        }

        return encoded;
    }
}
