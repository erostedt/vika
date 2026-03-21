cmake_minimum_required(VERSION 3.25)

project(vika LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS
    "-Wall -pedantic -Wextra -Werror -Wshadow -Wsign-conversion -Wconversion -Wunreachable-code -g")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_subdirectory(tests)
add_subdirectory(examples)
