# See apt.kitware.com for updated CMake versions on Ubuntu.
cmake_minimum_required(VERSION 3.30.4)

project(NVBenchDemo CUDA)

set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --display_error_number -allow-unsupported-compiler")

# Use CPM to fetch NVBench's main branch.
# CPM: https://github.com/cpm-cmake/CPM.cmake
# NVBench: https://github.com/NVIDIA/nvbench
include(cmake/CPM.cmake)
CPMAddPackage("gh:NVIDIA/nvbench#main")

add_executable(vector_add_bench day_046_nvbench_vector_add.cu)
target_link_libraries(vector_add_bench nvbench::main)

add_executable(prefix_sum_bench day_047_prefix_sum_bench_main.cu)
target_link_libraries(prefix_sum_bench nvbench::main)

add_executable(rsort day_045_radix_sort_finally_main.cu)

# this needs to come after add_executable
target_compile_options(rsort PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wextra,-Wno-unused-function>
)
