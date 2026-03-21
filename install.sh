#!/bin/bash

# Check if CMakeLists.txt exists
if [ ! -f "CMakeLists.txt" ]; then
    echo "CMakeLists.txt not found. Exiting..."
    exit 1
fi

# Set the build type based on the passed argument
BUILD_TYPE="Debug" # Default build type
if [ "$1" == "Release" ]; then
    BUILD_TYPE="Release"
fi

# Create a build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
fi

# Navigate to the build directory
cd build

# Run CMake to configure the project with the specified build type
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

# Run make to build the project
make

# Check if make was successful
if [ $? -eq 0 ]; then
    echo "Build successful in $BUILD_TYPE mode!"
else
    echo "Build failed in $BUILD_TYPE mode!"
fi

if [ -f "../compile_commands.json" ]; then
    rm ../compile_commands.json
fi

ln compile_commands.json ..
