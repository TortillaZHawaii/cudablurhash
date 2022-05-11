#include "../../include/csv/csv_logger.cuh"

namespace CuBlurHash
{
    CsvLogger::CsvLogger(std::string const& filename, std::string const& separator) : separator(separator)
    {
        csv_file.open(filename, std::ios_base::app);
    }

    CsvLogger::~CsvLogger()
    {
        csv_file.close();
    }

    void CsvLogger::log_encoding_results(std::string const& imgPath,
        Image const& image,
        std::string const& result,
        std::string const& version,
        int x_components, int y_components,
        int total_encoding_time_in_ms
        )
    {
        csv_file << imgPath << separator
            << result << separator
            << image.get_width() << separator
            << image.get_height() << separator
            << version << separator
            << x_components << separator << y_components << separator
            << "encoder" << separator
            << total_encoding_time_in_ms << std::endl;
    }
}