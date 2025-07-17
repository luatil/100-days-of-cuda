
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <vector>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

void vector_add_benchmark(nvbench::state &state)
{
    const auto n = state.get_int64("Elements");
    const auto threads_per_block = state.get_int64("Threads Per Block");
    const size_t bytes = n * sizeof(float);

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);

    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    state.exec([&](nvbench::launch &launch) {
        vector_add_kernel<<<blocks, threads_per_block, 0, launch.get_stream()>>>(d_a, d_b, d_c, n);
    });

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

NVBENCH_BENCH(vector_add_benchmark)
    .add_int64_power_of_two_axis("Elements", nvbench::range(10, 24, 1))
    .add_int64_power_of_two_axis("Threads Per Block", nvbench::range(7, 10, 1))
    .set_timeout(10);
