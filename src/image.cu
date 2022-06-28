#include "../include/image.cuh"

namespace CuBlurHash
{
    Image::Image(std::string const& filename)
    {
        FILE *fp = fopen(filename.c_str(), "rb");
        
        if(!fp)
        {
            printf("Couldn't open file.");
            exit(EXIT_FAILURE);
        }

        // read image data from file using libpng
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

        if(!png)
        {
            printf("Couldn't create png read struct.");
            exit(EXIT_FAILURE);
        }

        png_infop info = png_create_info_struct(png);
        if(!info)
        {
            printf("Couldn't create png info struct.");
            exit(EXIT_FAILURE);
        }

        if(setjmp(png_jmpbuf(png)))
        {
            printf("Error during png read.");
            exit(EXIT_FAILURE);
        }

        png_init_io(png, fp);

        png_read_info(png, info);

        width = png_get_image_width(png, info);
        height = png_get_image_height(png, info);
        channels = png_get_channels(png, info);
        auto color_type = png_get_color_type(png, info);
        auto bit_depth = png_get_bit_depth(png, info);

        if(bit_depth == 16)
        {
            png_set_strip_16(png);
        }

        if(color_type == PNG_COLOR_TYPE_PALETTE)
        {
            png_set_palette_to_rgb(png);
        }

        if(png_get_valid(png, info, PNG_INFO_tRNS))
        {
            png_set_tRNS_to_alpha(png);
        }

        // These color_type don't have an alpha channel then fill it with 0xff.
        // if(color_type == PNG_COLOR_TYPE_RGB ||
        //     color_type == PNG_COLOR_TYPE_GRAY ||
        //     color_type == PNG_COLOR_TYPE_PALETTE)
        //     png_set_filler(png, 0xFF, PNG_FILLER_AFTER);

        // if(color_type == PNG_COLOR_TYPE_GRAY ||
        //     color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        //     png_set_gray_to_rgb(png);

        png_read_update_info(png, info);

        auto row_pointers = std::vector<png_bytep>(height);
        for(auto i = 0; i < height; i++)
        {
            row_pointers[i] = (png_bytep)png_malloc(png, png_get_rowbytes(png, info));
        }

        png_read_image(png, row_pointers.data());
        
        // get pixels from row_pointers
        pixels = thrust::host_vector<RGBXY>(width * height);

        for(int y = 0; y < height; ++y)
        {
            auto row = row_pointers[y];
            for(int x = 0; x < width; ++x)
            {
                auto pixel = &row[x * channels];
                auto rgb = RGBi(pixel[0], pixel[1], pixel[2]);
                pixels[y * width + x] = RGBXY(rgb, x, y);
            }
        }

        for(auto i = 0; i < height; i++)
        {
            png_free(png, row_pointers[i]);
        }

        fclose(fp);
    }

    thrust::host_vector<RGBXY> Image::get_pixels() const
    {
        return pixels;
    }

    int Image::get_width() const
    {
        return width;
    }

    int Image::get_height() const
    {
        return height;
    }
}