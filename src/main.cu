#include <iostream>

#include "../include/encodes/encode.cuh"
#include "../include/image.cuh"

int main(int argc, const char **argv)
{
    // if(argc != 4 && argc != 5)
    // {
	// 	std::cerr << "Usage: " << argv[0] << " x_components y_components imagefile [csvfile]" << std::endl;
	// 	return 1;
	// }

    int x_components = 8;//atoi(argv[1]);
    int y_components = 8;//atoi(argv[2]);

    std::cout << "Loading image..." << std::endl;

    const char *filename = "../data/Lenna.png";//argv[3];
    CuBlurHash::Image image = CuBlurHash::Image(filename);

    std::cout << "Image loaded." << std::endl;

    std::cout << "Encoding image..." << std::endl;

    std::string hash = CuBlurHash::encode_image(image, x_components, y_components);

    std::cout << "Image encoded." << std::endl;

    std::cout << "Hash: " << std::endl;
    std::cout << hash << std::endl;

    return 0;
}
