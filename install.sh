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
    mkdir build || exit 1
fi

# Navigate to the build directory
cd build || exit 1

# Run CMake to configure the project with the specified build type. A failed configure used to
# fall through to make, which then failed again for a reason that had nothing to do with the code.
if ! cmake -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ..; then
    echo "Configure failed in $BUILD_TYPE mode!"
    exit 1
fi

# Run make to build the project
make
BUILD_STATUS=$?

# Refreshed either way: cmake rewrites compile_commands.json during configure, so leaving the old
# link behind after a failed build points an editor at a file that no longer matches.
if [ -f "../compile_commands.json" ]; then
    rm ../compile_commands.json
fi
ln compile_commands.json ..

# Exit with the build's status, not the link's. This used to end on `ln`, so a failed build printed
# "Build failed" and still exited 0 - and `./install.sh && ./build/tests/run_tests` ran the tests
# against whatever binary was there before.
if [ $BUILD_STATUS -eq 0 ]; then
    echo "Build successful in $BUILD_TYPE mode!"
else
    echo "Build failed in $BUILD_TYPE mode!"
fi
exit $BUILD_STATUS
