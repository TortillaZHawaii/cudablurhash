#include <iostream>
#include <cstdio>
#include <ctime>

#include "../include/csv/csv_logger.cuh"
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

    std::clock_t start_time = std::clock();

    std::string hash = CuBlurHash::encode_image(image, x_components, y_components);

    std::clock_t end_time = std::clock();

    int duration_ms = (end_time - start_time) * 1000 / CLOCKS_PER_SEC;

    std::cout << "Image encoded in " << duration_ms << " ms." << std::endl;

    if(argc == 5) 
    {
        std::string csv_file_path = std::string(argv[4]);
        std::cout << "Writing encoder logs to " << csv_file_path << std::endl;

        CuBlurHash::CsvLogger(csv_file_path, ">")
            .log_encoding_results(std::string(filename), image, hash,
                "Thrusted", x_components, y_components, duration_ms);
    }

    std::cout << "Hash: " << std::endl;
    std::cout << hash << std::endl;

    return 0;
}
