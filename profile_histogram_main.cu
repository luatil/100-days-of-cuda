#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define BLOCK_SIZE 256

// Histogram kernel 1: Single thread (from day_023_histogram_00.cu)
__global__ void HistogramSingleThread(const int *input, int *histogram, int N, int num_bins)
{
    for (int i = 0; i < N; i++)
    {
        int bin = input[i];
        histogram[bin]++;
    }
}

// Histogram kernel 2: Parallel with atomic operations (from day_023_histogram_01.cu)
__global__ void HistogramAtomic(const int *input, int *histogram, int N, int num_bins)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
    {
        atomicAdd(&histogram[input[tid]], 1);
    }
}

// Histogram kernel 3: Shared memory optimization
__global__ void HistogramShared(const int *input, int *histogram, int N, int num_bins)
{
    extern __shared__ int shared_hist[];
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        shared_hist[i] = 0;
    }
    __syncthreads();

    // Accumulate in shared memory
    if (tid < N)
    {
        atomicAdd(&shared_hist[input[tid]], 1);
    }
    __syncthreads();

    // Write back to global memory
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x)
    {
        if (shared_hist[i] > 0)
        {
            atomicAdd(&histogram[i], shared_hist[i]);
        }
    }
}

// CUDA timing utility
struct CUDATimer {
    cudaEvent_t start, stop;
    
    CUDATimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() {
        cudaEventRecord(start);
    }
    
    float stopTimer() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

void runProfile(const char* name, void (*kernel)(const int*, int*, int, int), 
                const int* d_input, int* d_histogram, int N, int num_bins, 
                int grid_size, int block_size, int shared_mem_size = 0)
{
    CUDATimer timer;
    
    // Reset histogram
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));
    
    // Warm up run
    if (shared_mem_size > 0) {
        kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, d_histogram, N, num_bins);
    } else {
        kernel<<<grid_size, block_size>>>(d_input, d_histogram, N, num_bins);
    }
    cudaDeviceSynchronize();
    
    // Timed runs
    const int num_runs = 10;
    float total_time = 0;
    
    for (int run = 0; run < num_runs; run++) {
        cudaMemset(d_histogram, 0, num_bins * sizeof(int));
        
        timer.startTimer();
        if (shared_mem_size > 0) {
            kernel<<<grid_size, block_size, shared_mem_size>>>(d_input, d_histogram, N, num_bins);
        } else {
            kernel<<<grid_size, block_size>>>(d_input, d_histogram, N, num_bins);
        }
        float time = timer.stopTimer();
        total_time += time;
    }
    
    float avg_time = total_time / num_runs;
    float throughput = (N / avg_time) / 1e6; // Million elements per second
    
    printf("%-25s: %8.3f ms | %8.2f M elements/s\n", name, avg_time, throughput);
}

int main(int argc, char** argv)
{
    // Test parameters
    int N = (argc > 1) ? atoi(argv[1]) : 10000000; // 10M elements by default
    int num_bins = (argc > 2) ? atoi(argv[2]) : 256;
    
    printf("Profiling histogram kernels:\n");
    printf("Elements: %d, Bins: %d\n", N, num_bins);
    printf("%-25s: %8s | %s\n", "Kernel", "Time", "Throughput");
    printf("------------------------------------------------\n");
    
    // Generate random input data
    int* h_input = (int*)malloc(N * sizeof(int));
    srand(42); // Fixed seed for reproducibility
    for (int i = 0; i < N; i++) {
        h_input[i] = rand() % num_bins;
    }
    
    // Allocate device memory
    int *d_input, *d_histogram;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_histogram, num_bins * sizeof(int));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Profile kernels
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Kernel 1: Single thread
    runProfile("Single Thread", HistogramSingleThread, d_input, d_histogram, N, num_bins, 1, 1);
    
    // Kernel 2: Parallel with atomic operations
    runProfile("Parallel Atomic", HistogramAtomic, d_input, d_histogram, N, num_bins, grid_size, BLOCK_SIZE);
    
    // Kernel 3: Shared memory optimization
    int shared_mem_size = num_bins * sizeof(int);
    runProfile("Shared Memory", HistogramShared, d_input, d_histogram, N, num_bins, grid_size, BLOCK_SIZE, shared_mem_size);
    
    // Verify correctness with CPU reference
    int* h_histogram_ref = (int*)calloc(num_bins, sizeof(int));
    for (int i = 0; i < N; i++) {
        h_histogram_ref[h_input[i]]++;
    }
    
    // Check GPU result
    int* h_histogram_gpu = (int*)malloc(num_bins * sizeof(int));
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));
    HistogramAtomic<<<grid_size, BLOCK_SIZE>>>(d_input, d_histogram, N, num_bins);
    cudaMemcpy(h_histogram_gpu, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    bool correct = true;
    for (int i = 0; i < num_bins; i++) {
        if (h_histogram_ref[i] != h_histogram_gpu[i]) {
            correct = false;
            break;
        }
    }
    
    printf("\nVerification: %s\n", correct ? "PASSED" : "FAILED");
    
    // Cleanup
    free(h_input);
    free(h_histogram_ref);
    free(h_histogram_gpu);
    cudaFree(d_input);
    cudaFree(d_histogram);
    
    return 0;
}