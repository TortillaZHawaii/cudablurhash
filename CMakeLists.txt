cmake_minimum_required(VERSION 3.7)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(cublurhash LANGUAGES CXX CUDA)

add_executable(
  cublurhash
  src/main.cu
  src/utils.cu
  src/image.cu
  src/encodes/encode.cu
  )

target_compile_features(cublurhash PUBLIC cxx_std_14)
set_target_properties(cublurhash PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                          CUDA_ARCHITECTURES OFF)

if(APPLE)
  set_property(TARGET cublurhash PROPERTY BUILD_RPATH
                                        ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()
