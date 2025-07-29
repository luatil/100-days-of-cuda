# See apt.kitware.com for updated CMake versions on Ubuntu.
cmake_minimum_required(VERSION 3.30.4)

project(100DaysOfCuda CUDA)

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

add_executable(reduction_bench day_048_reduction_benchmark_main.cu)
target_link_libraries(reduction_bench nvbench::main)

add_executable(test_bench day_059_convolution_benchmark.cu)
target_link_libraries(test_bench nvbench::main)

add_executable(rsort day_045_radix_sort_finally_main.cu)

# this needs to come after add_executable
target_compile_options(rsort PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wextra,-Wno-unused-function>
)

add_executable(example day_049_something_main.cu)
target_compile_options(example PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Werror,-Wextra,-Wno-unused-function>
)

# Set default install prefix to ~/.local if not specified
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "$ENV{HOME}/.local" CACHE PATH "Install prefix" FORCE)
endif()

# Install all executables
install(TARGETS vector_add_bench prefix_sum_bench reduction_bench rsort example
    RUNTIME DESTINATION bin
)
