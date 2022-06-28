cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(cublurhash LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# search for pkg-config
include (FindPkgConfig)
if (NOT PKG_CONFIG_FOUND)
message (FATAL_ERROR "pkg-config not found")
endif ()

# check for libpng
pkg_check_modules (LIBPNG libpng16 REQUIRED)
if (NOT LIBPNG_FOUND)
message(FATAL_ERROR "You don't seem to have libpng16 development libraries installed")
else ()
include_directories (${LIBPNG_INCLUDE_DIRS})
link_directories (${LIBPNG_LIBRARY_DIRS})
link_libraries (${LIBPNG_LIBRARIES})
endif ()
add_executable(
  cublurhash
  src/main.cu
  src/utils.cu
  src/image.cu
  src/encodes/encode.cu
  src/csv/csv_logger.cu
  ${LIBPNG_LINK_FLAGS}
  )
set_target_properties(cublurhash PROPERTIES CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES OFF)

if(APPLE)
  set_property(TARGET cublurhash PROPERTY BUILD_RPATH
                                        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
