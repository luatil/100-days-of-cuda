#include <chrono>
#include <cuda_runtime.h>
#include <math.h>
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

// Fixed verification function - using fabsf() instead of abs()
bool VerifyResults(float *HA, float *HB, float *HResult, int N, const char *MethodName)
{
    bool Correct = true;
    const float EPSILON = 1e-5f;
    int ErrorCount = 0;

    printf("Verifying %s results...\n", MethodName);

    for (int I = 0; I < N; I++)
    {
        // Expected: ((a + b) * 2.0)^2
        float Expected = (HA[I] + HB[I]) * 2.0f;
        Expected = Expected * Expected;

        // BUG FIX: Use fabsf() instead of abs() for floating point comparison
        if (fabsf(HResult[I] - Expected) > EPSILON)
        {
            if (ErrorCount < 10) // Limit error reporting
            {
                printf("  Mismatch at index %d: got %.6f, expected %.6f, diff: %.6f\n", I, HResult[I], Expected,
                       fabsf(HResult[I] - Expected));
            }
            Correct = false;
            ErrorCount++;
        }
    }

    if (ErrorCount > 0)
    {
        printf("  Total errors found: %d out of %d elements\n", ErrorCount, N);
    }

    return Correct;
}

// Compare two result arrays with detailed analysis
bool CompareResults(float *Result1, float *Result2, int N, const char *Name1, const char *Name2)
{
    bool Match = true;
    const float EPSILON = 1e-5f;
    int MismatchCount = 0;
    float MaxDiff = 0.0f;

    printf("Comparing %s vs %s results...\n", Name1, Name2);

    for (int I = 0; I < N; I++)
    {
        // BUG FIX: Use fabsf() instead of abs() for floating point comparison
        float Diff = fabsf(Result1[I] - Result2[I]);

        if (Diff > MaxDiff)
        {
            MaxDiff = Diff;
        }

        if (Diff > EPSILON)
        {
            if (MismatchCount < 10) // Limit error reporting
            {
                printf("  Mismatch at index %d: %s=%.6f, %s=%.6f, diff=%.6f\n", I, Name1, Result1[I], Name2, Result2[I],
                       Diff);
            }
            Match = false;
            MismatchCount++;
        }
    }

    printf("  Maximum difference: %.6f\n", MaxDiff);
    if (MismatchCount > 0)
    {
        printf("  Total mismatches: %d out of %d elements\n", MismatchCount, N);
    }

    return Match;
}

// Debug function to print memory state
void DebugMemoryState(float *HA, float *HB, float *HResult, int StartIdx, int Count, const char *Label)
{
    printf("\n%s - Memory State (elements %d to %d):\n", Label, StartIdx, StartIdx + Count - 1);
    printf("Index |   A     |   B     | Result  | Expected\n");
    printf("------|---------|---------|---------|----------\n");

    for (int I = StartIdx; I < StartIdx + Count && I < VECTOR_SIZE; I++)
    {
        float Expected = (HA[I] + HB[I]) * 2.0f;
        Expected = Expected * Expected;
        printf("%5d | %7.6f | %7.6f | %7.6f | %8.6f\n", I, HA[I], HB[I], HResult[I], Expected);
    }
}

int main()
{
    printf("CUDA Graphs Bug Investigation - Day 028\n");
    printf("Investigating potential bugs in day_027 implementation\n");
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

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Traditional launches: %8.2f ms\n", TimeTraditional);
    printf("CUDA Graphs:         %8.2f ms\n", TimeGraph);
    printf("Speedup:             %8.2fx\n", TimeTraditional / TimeGraph);
    printf("Overhead reduction:  %8.2f%%\n", 100.0f * (TimeTraditional - TimeGraph) / TimeTraditional);

    // Verify correctness with fixed verification function
    printf("\nDetailed Verification (FIXED):\n");
    bool TraditionalCorrect = VerifyResults(HA, HB, HResultTraditional, VECTOR_SIZE, "Traditional");
    bool GraphCorrect = VerifyResults(HA, HB, HResultGraph, VECTOR_SIZE, "Graph");

    printf("\nFinal Results:\n");
    printf("Traditional results: %s\n", TraditionalCorrect ? "CORRECT" : "INCORRECT");
    printf("Graph results:       %s\n", GraphCorrect ? "CORRECT" : "INCORRECT");

    // Compare results between methods with fixed comparison
    printf("\nResult Comparison (FIXED):\n");
    bool ResultsMatch = CompareResults(HResultTraditional, HResultGraph, VECTOR_SIZE, "Traditional", "Graph");
    printf("Results match:       %s\n", ResultsMatch ? "YES" : "NO");

    // Debug memory state for first few elements
    DebugMemoryState(HA, HB, HResultTraditional, 0, 5, "Traditional Results");
    DebugMemoryState(HA, HB, HResultGraph, 0, 5, "Graph Results");

    // Check for edge cases - test with some specific indices
    printf("\nEdge Case Analysis:\n");
    int TestIndices[] = {0, 1, VECTOR_SIZE / 2, VECTOR_SIZE - 2, VECTOR_SIZE - 1};
    int NumTestIndices = sizeof(TestIndices) / sizeof(TestIndices[0]);

    for (int I = 0; I < NumTestIndices; I++)
    {
        int Idx = TestIndices[I];
        if (Idx < VECTOR_SIZE)
        {
            float Expected = (HA[Idx] + HB[Idx]) * 2.0f;
            Expected = Expected * Expected;
            printf("Index %d: Traditional=%.6f, Graph=%.6f, Expected=%.6f\n", Idx, HResultTraditional[Idx],
                   HResultGraph[Idx], Expected);
        }
    }

    printf("\nBug Analysis Summary:\n");
    printf("1. Fixed abs() -> fabsf() for float comparisons\n");
    printf("2. Added detailed error reporting and analysis\n");
    printf("3. Implemented comprehensive result comparison\n");
    printf("4. Added edge case testing\n");

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
