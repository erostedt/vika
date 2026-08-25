cmake_minimum_required(VERSION 3.25)

project(vika LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Every source file in this project is .cu, so CMAKE_CXX_FLAGS reaches none of them - the host
# compiler only ever sees these through nvcc's -Xcompiler, which is why this list had been
# inert. Comma-separated: -Xcompiler takes one argument.
#
# -pedantic is deliberately absent. It reports 7168 warnings here, and none of them are
# actionable: 2767 are nvcc's own generated #line directives, and the rest are three extensions
# this codebase uses on purpose - statement expressions in VIKA_UNWRAP_OR_RETURN, C++20
# designated initializers at C++17, and ##__VA_ARGS__ in VIKA_PANIC.
set(VIKA_WARNINGS "-Wall,-Wextra,-Werror,-Wshadow,-Wsign-conversion,-Wconversion,-Wunreachable-code")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=${VIKA_WARNINGS}")

set(BUILD_PATH ${CMAKE_CURRENT_SOURCE_DIR}/build)

add_subdirectory(tests)
add_subdirectory(examples)
