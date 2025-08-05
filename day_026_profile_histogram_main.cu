#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

// Histogram kernel 1: Single thread (from day_023_histogram_00.cu)
__global__ void HistogramSingleThread(const int *Input, int *Histogram, int N, int NumBins)
{
    for (int I = 0; I < N; I++)
    {
        int Bin = Input[I];
        Histogram[Bin]++;
    }
}

// Histogram kernel 2: Parallel with atomic operations (from day_023_histogram_01.cu)
__global__ void HistogramAtomic(const int *Input, int *Histogram, int N, int NumBins)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        atomicAdd(&Histogram[Input[Tid]], 1);
    }
}

// Histogram kernel 3: Shared memory optimization
__global__ void HistogramShared(const int *Input, int *Histogram, int N, int NumBins)
{
    extern __shared__ int SharedHist[];
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    // Initialize shared memory
    for (int I = threadIdx.x; I < NumBins; I += blockDim.x)
    {
        SharedHist[I] = 0;
    }
    __syncthreads();

    // Accumulate in shared memory
    if (Tid < N)
    {
        atomicAdd(&SharedHist[Input[Tid]], 1);
    }
    __syncthreads();

    // Write back to global memory
    for (int I = threadIdx.x; I < NumBins; I += blockDim.x)
    {
        if (SharedHist[I] > 0)
        {
            atomicAdd(&Histogram[I], SharedHist[I]);
        }
    }
}

// CUDA timing utility
struct cuda_timer
{
    cudaEvent_t Start, Stop;

    cuda_timer()
    {
        cudaEventCreate(&Start);
        cudaEventCreate(&Stop);
    }

    ~cuda_timer()
    {
        cudaEventDestroy(Start);
        cudaEventDestroy(Stop);
    }

    void StartTimer()
    {
        cudaEventRecord(Start);
    }

    float StopTimer()
    {
        cudaEventRecord(Stop);
        cudaEventSynchronize(Stop);
        float Milliseconds = 0;
        cudaEventElapsedTime(&Milliseconds, Start, Stop);
        return Milliseconds;
    }
};

void RunProfile(const char *Name, void (*Kernel)(const int *, int *, int, int), const int *DInput, int *DHistogram,
                int N, int NumBins, int GridSize, int BlockSize, int SharedMemSize = 0)
{
    cuda_timer Timer;

    // Reset histogram
    cudaMemset(DHistogram, 0, NumBins * sizeof(int));

    // Warm up run
    if (SharedMemSize > 0)
    {
        Kernel<<<GridSize, BlockSize, SharedMemSize>>>(DInput, DHistogram, N, NumBins);
    }
    else
    {
        Kernel<<<GridSize, BlockSize>>>(DInput, DHistogram, N, NumBins);
    }
    cudaDeviceSynchronize();

    // Timed runs
    const int NUM_RUNS = 10;
    float TotalTime = 0;

    for (int Run = 0; Run < NUM_RUNS; Run++)
    {
        cudaMemset(DHistogram, 0, NumBins * sizeof(int));

        Timer.StartTimer();
        if (SharedMemSize > 0)
        {
            Kernel<<<GridSize, BlockSize, SharedMemSize>>>(DInput, DHistogram, N, NumBins);
        }
        else
        {
            Kernel<<<GridSize, BlockSize>>>(DInput, DHistogram, N, NumBins);
        }
        float Time = Timer.StopTimer();
        TotalTime += Time;
    }

    float AvgTime = TotalTime / NUM_RUNS;
    float Throughput = (N / AvgTime) / 1e6; // Million elements per second

    printf("%-25s: %8.3f ms | %8.2f M elements/s\n", Name, AvgTime, Throughput);
}

int main(int argc, char **argv)
{
    // Test parameters
    int N = (argc > 1) ? atoi(argv[1]) : 10000000; // 10M elements by default
    int NumBins = (argc > 2) ? atoi(argv[2]) : 256;

    printf("Profiling histogram kernels:\n");
    printf("Elements: %d, Bins: %d\n", N, NumBins);
    printf("%-25s: %8s | %s\n", "Kernel", "Time", "Throughput");
    printf("------------------------------------------------\n");

    // Generate random input data
    int *HInput = (int *)malloc(N * sizeof(int));
    srand(42); // Fixed seed for reproducibility
    for (int I = 0; I < N; I++)
    {
        HInput[I] = rand() % NumBins;
    }

    // Allocate device memory
    int *DInput, *DHistogram;
    cudaMalloc(&DInput, N * sizeof(int));
    cudaMalloc(&DHistogram, NumBins * sizeof(int));

    // Copy input to device
    cudaMemcpy(DInput, HInput, N * sizeof(int), cudaMemcpyHostToDevice);

    // Profile kernels
    int GridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Kernel 1: Single thread
    RunProfile("Single Thread", HistogramSingleThread, DInput, DHistogram, N, NumBins, 1, 1);

    // Kernel 2: Parallel with atomic operations
    RunProfile("Parallel Atomic", HistogramAtomic, DInput, DHistogram, N, NumBins, GridSize, BLOCK_SIZE);

    // Kernel 3: Shared memory optimization
    int SharedMemSize = NumBins * sizeof(int);
    RunProfile("Shared Memory", HistogramShared, DInput, DHistogram, N, NumBins, GridSize, BLOCK_SIZE,
               SharedMemSize);

    // Verify correctness with CPU reference
    int *HHistogramRef = (int *)calloc(NumBins, sizeof(int));
    for (int I = 0; I < N; I++)
    {
        HHistogramRef[HInput[I]]++;
    }

    // Check GPU result
    int *HHistogramGpu = (int *)malloc(NumBins * sizeof(int));
    cudaMemset(DHistogram, 0, NumBins * sizeof(int));
    HistogramAtomic<<<GridSize, BLOCK_SIZE>>>(DInput, DHistogram, N, NumBins);
    cudaMemcpy(HHistogramGpu, DHistogram, NumBins * sizeof(int), cudaMemcpyDeviceToHost);

    bool Correct = true;
    for (int I = 0; I < NumBins; I++)
    {
        if (HHistogramRef[I] != HHistogramGpu[I])
        {
            Correct = false;
            break;
        }
    }

    printf("\nVerification: %s\n", Correct ? "PASSED" : "FAILED");

    // Cleanup
    free(HInput);
    free(HHistogramRef);
    free(HHistogramGpu);
    cudaFree(DInput);
    cudaFree(DHistogram);

    return 0;
}