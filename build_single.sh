#!/bin/bash

# Check if filename provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <source_file>"
    exit 1
fi

# Create build directory if it doesn't exist
mkdir -p build
pushd build

# Get filename without extension
filename=$(basename "$1" .cpp)

# Check for and use nvcc if available
if command -v nvcc &> /dev/null; then
    echo "Building with nvcc..."

    # Check if this is a matgen file that needs curand
    EXTRA_LIBS=""
    if [[ "$1" == *"matgen"* ]]; then
        EXTRA_LIBS="-lcurand"
    fi

    # Debug build
    nvcc -DDEBUG_ENABLED=1 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 "../$1" -o "${filename}_dn" $EXTRA_LIBS

    # Release build
    nvcc -DDEBUG_ENABLED=0 -O3 -g -Xcompiler "-Wall -Werror -Wextra -Wno-unused-function" -Xcudafe --display_error_number -allow-unsupported-compiler -arch=sm_86 -gencode=arch=compute_86,code=sm_86 "../$1" -o "${filename}_rn" $EXTRA_LIBS
else
    echo "nvcc not found - Skipping builds with nvcc"
fi

popd
