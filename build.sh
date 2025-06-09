#!/bin/bash

# Check for compilers and warn if missing
if ! command -v nvcc &> /dev/null; then
    echo "WARNING: nvcc not found -- NVCC executable will not be built"
fi

# Set up pattern matching
if [ $# -eq 0 ]; then
    buildpat="*_main.cu"
else
    buildpat="*$1*_main.cu"
fi

# Find and build matching files
for file in $buildpat; do
    # Check if file actually exists (handles case where no matches found)
    if [ -f "$file" ]; then
        echo "Building $file..."
        ./build_single.sh "$file"
    fi
done

# Handle case where no files match the pattern
if ! ls $buildpat 1> /dev/null 2>&1; then
    echo "No files matching pattern '$buildpat' found"
fi
