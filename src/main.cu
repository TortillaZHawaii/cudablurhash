#include <iostream>

#include "../include/encodes/encode.cuh"
#include "../include/image.cuh"

int main(int argc, const char **argv)
{
    if(argc != 4 && argc != 5)
    {
		std::cerr << "Usage: " << argv[0] << " x_components y_components imagefile [csvfile]" << std::endl;
		return 1;
	}

    int x_components = atoi(argv[1]);
    int y_components = atoi(argv[2]);

    if(x_components > 9 || x_components < 1)
    {
        std::cerr << "x_components must be between 1 and 9" << std::endl;
        return 1;
    }

    if(y_components > 9 || y_components < 1)
    {
        std::cerr << "y_components must be between 1 and 9" << std::endl;
        return 1;
    }

    std::cout << "Loading image..." << std::endl;

    const char *filename = argv[3];
    CuBlurHash::Image image = CuBlurHash::Image(filename);

    std::cout << "Image loaded." << std::endl;

    std::cout << "Encoding image..." << std::endl;

    std::string hash = CuBlurHash::encode_image(image, x_components, y_components);

    std::cout << "Image encoded." << std::endl;

    std::cout << "Hash: " << std::endl;
    std::cout << hash << std::endl;

    return 0;
}
