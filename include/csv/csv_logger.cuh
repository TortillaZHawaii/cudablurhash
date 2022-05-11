#pragma once

#include <fstream>
#include "../image.cuh"

namespace CuBlurHash
{
    class CsvLogger
    {
        std::ofstream csv_file;
        std::string separator;

        public:
        CsvLogger(std::string const& filename, std::string const& separator);
        ~CsvLogger();

        void log_encoding_results(std::string const& imgPath,
            Image const& image,
            std::string const& result,
            std::string const& version,
            int x_components, int y_components,
            int total_encoding_time_in_ms
            );
    };
}