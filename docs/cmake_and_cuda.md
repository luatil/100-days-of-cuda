# cuda-with-cmake

An example repository demonstrating how to build CUDA projects with CMake and configure GitHub Actions for automated testing.

## Overview

This project contains a simple CUDA vector addition example that showcases:
- Building CUDA applications with CMake
- Configuring proper compiler paths for CUDA development
- Setting up GitHub Actions CI with self-hosted GPU runners
- Automated testing of CUDA kernels

## Project Structure

```
.
├── vector_add.cu              # CUDA vector addition kernel and test
├── CMakeLists.txt             # CMake configuration
└── .github/workflows/
    └── cuda-test.yml          # GitHub Actions workflow
```

## Vector Addition Example

The `vector_add.cu` program performs element-wise addition of two vectors on the GPU:
- Allocates device memory
- Copies input vectors to GPU
- Launches CUDA kernel for parallel addition
- Verifies results (exits with code 0 on success, 1 on failure)

## Building

### Prerequisites
- CUDA Toolkit (nvcc compiler)
- CMake 4.1+
- GCC 13 (or compatible version)

### Build Steps

```bash
mkdir build
cd build
cmake .. \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc
cmake --build .
```

### Running the Test

```bash
./build/vector_add
```

Expected output: `VECTOR_ADD: Pass`

## GitHub Actions Configuration

### Self-Hosted Runner Setup

This project uses a self-hosted GitHub Actions runner with GPU support. The workflow (`.github/workflows/cuda-test.yml`) automatically:

1. Checks out the code
2. Configures the project with CMake
3. Builds the CUDA executable
4. Runs the vector addition test on actual GPU hardware

### Workflow Trigger

The workflow runs on:
- Push to any branch
- Pull requests

### Why Self-Hosted?

GitHub's cloud runners don't have NVIDIA GPUs. To test CUDA code, you need:
- A self-hosted runner with NVIDIA GPU
- CUDA Toolkit installed on the runner
- Proper GPU drivers configured

## CMake Configuration Details

The `CMakeLists.txt` specifies:
- **Compiler paths**: Ensures CMake uses the correct GCC and NVCC versions
- **CUDA architectures**: Set to `native` to compile for the local GPU
- **CUDA flags**: Includes line info for debugging and suppresses deprecated GPU warnings

## References

- [GitHub Actions with GPU](https://betatim.github.io/posts/github-action-with-gpu/)
- [CMake CUDA Support](https://cmake.org/cmake/help/latest/manual/cmake-language.7.html#cuda)
- [Self-Hosted GitHub Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
