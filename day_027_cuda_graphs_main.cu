#include <chrono>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define VECTOR_SIZE (1024 * 1024) // 1M elements
#define NUM_ITERATIONS 1000

// Simple vector addition kernel
__global__ void VectorAdd(const float *A, const float *B, float *C, int N)
{
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < N)
    {
        C[Idx] = A[Idx] + B[Idx];
    }
}

// Vector scaling kernel
__global__ void VectorScale(float *A, float Scale, int N)
{
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < N)
    {
        A[Idx] *= Scale;
    }
}

// Vector square kernel
__global__ void VectorSquare(float *A, int N)
{
    int Idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (Idx < N)
    {
        A[Idx] = A[Idx] * A[Idx];
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

// Traditional kernel launches
float RunTraditionalLaunches(float *DA, float *DB, float *DC, float *DTemp, int N)
{
    int GridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cuda_timer Timer;

    Timer.StartTimer();

    for (int I = 0; I < NUM_ITERATIONS; I++)
    {
        // Sequence of operations: add -> scale -> square
        VectorAdd<<<GridSize, BLOCK_SIZE>>>(DA, DB, DTemp, N);
        VectorScale<<<GridSize, BLOCK_SIZE>>>(DTemp, 2.0f, N);
        VectorSquare<<<GridSize, BLOCK_SIZE>>>(DTemp, N);

        // Copy result back to d_c
        cudaMemcpy(DC, DTemp, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    return Timer.StopTimer();
}

// CUDA Graphs execution
float RunCudaGraphs(float *DA, float *DB, float *DC, float *DTemp, int N)
{
    int GridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaGraph_t Graph;
    cudaGraphExec_t GraphExec;
    cudaStream_t Stream;

    // Create stream for graph capture
    cudaStreamCreate(&Stream);

    // Start graph capture
    cudaStreamBeginCapture(Stream, cudaStreamCaptureModeGlobal);

    // Record the sequence of operations
    VectorAdd<<<GridSize, BLOCK_SIZE, 0, Stream>>>(DA, DB, DTemp, N);
    VectorScale<<<GridSize, BLOCK_SIZE, 0, Stream>>>(DTemp, 2.0f, N);
    VectorSquare<<<GridSize, BLOCK_SIZE, 0, Stream>>>(DTemp, N);
    cudaMemcpyAsync(DC, DTemp, N * sizeof(float), cudaMemcpyDeviceToDevice, Stream);

    // End capture and create graph
    cudaStreamEndCapture(Stream, &Graph);

    // Instantiate the graph
    cudaGraphInstantiate(&GraphExec, Graph, NULL, NULL, 0);

    // Time the graph execution
    cuda_timer Timer;
    Timer.StartTimer();

    for (int I = 0; I < NUM_ITERATIONS; I++)
    {
        cudaGraphLaunch(GraphExec, Stream);
    }

    cudaStreamSynchronize(Stream);
    float Elapsed = Timer.StopTimer();

    // Cleanup
    cudaGraphExecDestroy(GraphExec);
    cudaGraphDestroy(Graph);
    cudaStreamDestroy(Stream);

    return Elapsed;
}

// Verify results are correct
bool VerifyResults(float *HA, float *HB, float *HResult, int N)
{
    bool Correct = true;
    const float EPSILON = 1e-5f;

    for (int I = 0; I < N; I++)
    {
        // Expected: ((a + b) * 2.0)^2
        float Expected = (HA[I] + HB[I]) * 2.0f;
        Expected = Expected * Expected;

        if (abs(HResult[I] - Expected) > EPSILON)
        {
            printf("Mismatch at index %d: got %f, expected %f\n", I, HResult[I], Expected);
            Correct = false;
            if (I > 10)
                break; // Don't spam too many errors
        }
    }

    return Correct;
}

int main()
{
    printf("CUDA Graphs Example - Day 027\n");
    printf("Vector size: %d elements\n", VECTOR_SIZE);
    printf("Iterations: %d\n", NUM_ITERATIONS);
    printf("============================================\n");

    // Allocate host memory
    float *HA = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *HB = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *HResultTraditional = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *HResultGraph = (float *)malloc(VECTOR_SIZE * sizeof(float));

    // Initialize input data
    for (int I = 0; I < VECTOR_SIZE; I++)
    {
        HA[I] = (float)I / 1000.0f;
        HB[I] = (float)(I + 1) / 1000.0f;
    }

    // Allocate device memory
    float *DA, *DB, *DC, *DTemp;
    cudaMalloc(&DA, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&DB, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&DC, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&DTemp, VECTOR_SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(DA, HA, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(DB, HB, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Run traditional kernel launches
    printf("Running traditional kernel launches...\n");
    float TimeTraditional = RunTraditionalLaunches(DA, DB, DC, DTemp, VECTOR_SIZE);
    cudaMemcpy(HResultTraditional, DC, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CUDA Graphs
    printf("Running CUDA Graphs...\n");
    float TimeGraph = RunCudaGraphs(DA, DB, DC, DTemp, VECTOR_SIZE);
    cudaMemcpy(HResultGraph, DC, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("\nPerformance Results:\n");
    printf("Traditional launches: %8.2f ms\n", TimeTraditional);
    printf("CUDA Graphs:         %8.2f ms\n", TimeGraph);
    printf("Speedup:             %8.2fx\n", TimeTraditional / TimeGraph);
    printf("Overhead reduction:  %8.2f%%\n", 100.0f * (TimeTraditional - TimeGraph) / TimeTraditional);

    // Verify correctness
    printf("\nVerification:\n");
    bool TraditionalCorrect = VerifyResults(HA, HB, HResultTraditional, VECTOR_SIZE);
    bool GraphCorrect = VerifyResults(HA, HB, HResultGraph, VECTOR_SIZE);

    printf("Traditional results: %s\n", TraditionalCorrect ? "CORRECT" : "INCORRECT");
    printf("Graph results:       %s\n", GraphCorrect ? "CORRECT" : "INCORRECT");

    // Check if both methods produce the same results
    bool ResultsMatch = true;
    for (int I = 0; I < VECTOR_SIZE; I++)
    {
        if (abs(HResultTraditional[I] - HResultGraph[I]) > 1e-5f)
        {
            ResultsMatch = false;
            break;
        }
    }
    printf("Results match:       %s\n", ResultsMatch ? "YES" : "NO");

    // Show example values
    printf("\nSample Results (first 5 elements):\n");
    printf("Index | Input A | Input B | Traditional | Graph\n");
    printf("------|---------|---------|-------------|-------\n");
    for (int I = 0; I < 5; I++)
    {
        printf("%5d | %7.3f | %7.3f | %11.3f | %7.3f\n", I, HA[I], HB[I], HResultTraditional[I], HResultGraph[I]);
    }

    // Cleanup
    free(HA);
    free(HB);
    free(HResultTraditional);
    free(HResultGraph);
    cudaFree(DA);
    cudaFree(DB);
    cudaFree(DC);
    cudaFree(DTemp);

    return 0;
}
