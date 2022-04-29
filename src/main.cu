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

    CuBlurHash::Image image = CuBlurHash::Image(argv[3]);

    std::string hash = CuBlurHash::encode_image(image, x_components, y_components);

    std::cout << hash << std::endl;

    return 0;
}
