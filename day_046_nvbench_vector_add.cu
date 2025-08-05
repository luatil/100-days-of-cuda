
#include <cuda_runtime.h>
#include <nvbench/nvbench.cuh>
#include <vector>

__global__ void VectorAddKernel(const float *A, const float *B, float *C, int N)
{
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < N)
    {
        C[Idx] = A[Idx] + B[Idx];
    }
}

void VectorAddBenchmark(nvbench::state &State)
{
    const auto N = state.get_int64("Elements");
    const auto THREADS_PER_BLOCK = state.get_int64("Threads Per Block");
    const size_t BYTES = n * sizeof(float);

    float *DA;
    float *DB;
    float *DC;

    cudaMalloc(&DA, BYTES);
    cudaMalloc(&DB, BYTES);
    cudaMalloc(&DC, BYTES);

    std::vector<float> HA(N, 1.0f);
    std::vector<float> HB(N, 2.0f);

    cudaMemcpy(DA, h_a.data(), BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(DB, h_b.data(), BYTES, cudaMemcpyHostToDevice);

    const int BLOCKS = (N + threads_per_block - 1) / threads_per_block;

    State.exec([&](nvbench::launch &Launch) {
        VectorAddKernel<<<BLOCKS, threads_per_block, 0, launch.get_stream()>>>(DA, DB, DC, n);
    });

    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
}

NVBENCH_BENCH(VectorAddBenchmark)
    .add_int64_power_of_two_axis("Elements", nvbench::range(10, 24, 1))
    .add_int64_power_of_two_axis("Threads Per Block", nvbench::range(7, 10, 1))
    .set_timeout(10);
