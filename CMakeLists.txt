cmake_minimum_required(VERSION 4.1)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-13")
set(CMAKE_CUDA_COMPILER "/opt/cuda/bin/nvcc")

project(100DaysOfCuda CUDA CXX)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number -allow-unsupported-compiler")

# Use CPM to fetch NVBench's main branch.
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
# include(cmake/CPM.cmake)
# CPMAddPackage("gh:NVIDIA/nvbench#main")

# Include FetchContent module for downloading dependencies
include(FetchContent)

# Fetch Catch2
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.5.2  # Use the latest stable version
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(CTest)
include(Catch)

add_executable(
  subarray_sum
  day_079_subarray_sum_main.cu
  day_079_subarray_sum_cpu_naive.cu
  day_079_subarray_sum_gpu_v1_naive.cu
)

# Create test executable
add_executable(unit_tests
  day_076_vector_add.cu
  day_076_test_vector_add.cu
  day_077_property_test.cu
  day_078_logistic_regression.cu
  day_078_logistic_regression_test.cu
  day_079_subarray_sum_cpu_naive.cu
  day_079_subarray_sum_gpu_v1_naive.cu
  day_079_subarray_sum_test.cu
)

# Link against Catch2 and your library
target_link_libraries(unit_tests PRIVATE
    Catch2::Catch2WithMain
)

# Discover tests for CTest
include(Catch)
catch_discover_tests(unit_tests)

# Benchmark executable
add_executable(subarray_sum_benchmark
  day_079_subarray_sum_benchmark.cu
  day_079_subarray_sum_cpu_naive.cu
  day_079_subarray_sum_gpu_v1_naive.cu
)

# add_executable(vector_add_bench day_046_nvbench_vector_add.cu)
# target_link_libraries(vector_add_bench nvbench::main)
# 
# add_executable(prefix_sum_bench day_047_prefix_sum_bench_main.cu)
# target_link_libraries(prefix_sum_bench nvbench::main)

# add_executable(reduction_bench day_048_reduction_benchmark_main.cu)
# target_link_libraries(reduction_bench nvbench::main)
# 
# add_executable(test_bench day_059_convolution_benchmark.cu)
# target_link_libraries(test_bench nvbench::main)
# 
# add_executable(rsort day_045_radix_sort_finally_main.cu)
# 
# # this needs to come after add_executable
# target_compile_options(rsort PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wextra,-Wno-unused-function>
# )
# 
# add_executable(example day_049_something_main.cu)
# target_compile_options(example PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wextra,-Wno-unused-function>
# )

# Set default install prefix to ~/.local if not specified
# if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
#     set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install prefix" FORCE)
# endif()

# Install all executables
# install(TARGETS vector_add_bench prefix_sum_bench reduction_bench rsort example
#     RUNTIME DESTINATION bin
# )
