#include <chrono>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256
#define VECTOR_SIZE (1024 * 1024) // 1M elements
#define NUM_ITERATIONS 1000

// Simple vector addition kernel
__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector scaling kernel
__global__ void vectorScale(float *a, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] *= scale;
    }
}

// Vector square kernel
__global__ void vectorSquare(float *a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        a[idx] = a[idx] * a[idx];
    }
}

// CUDA timing utility
struct CUDATimer
{
    cudaEvent_t start, stop;

    CUDATimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~CUDATimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void startTimer()
    {
        cudaEventRecord(start);
    }

    float stopTimer()
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds;
    }
};

// Traditional kernel launches
float runTraditionalLaunches(float *d_a, float *d_b, float *d_c, float *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDATimer timer;

    timer.startTimer();

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        // Sequence of operations: add -> scale -> square
        vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        vectorScale<<<gridSize, BLOCK_SIZE>>>(d_temp, 2.0f, n);
        vectorSquare<<<gridSize, BLOCK_SIZE>>>(d_temp, n);

        // Copy result back to d_c
        cudaMemcpy(d_c, d_temp, n * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();
    return timer.stopTimer();
}

// CUDA Graphs execution
float runCudaGraphs(float *d_a, float *d_b, float *d_c, float *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    // Create stream for graph capture
    cudaStreamCreate(&stream);

    // Start graph capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Record the sequence of operations
    vectorAdd<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_a, d_b, d_temp, n);
    vectorScale<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_temp, 2.0f, n);
    vectorSquare<<<gridSize, BLOCK_SIZE, 0, stream>>>(d_temp, n);
    cudaMemcpyAsync(d_c, d_temp, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // End capture and create graph
    cudaStreamEndCapture(stream, &graph);

    // Instantiate the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Time the graph execution
    CUDATimer timer;
    timer.startTimer();

    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        cudaGraphLaunch(graphExec, stream);
    }

    cudaStreamSynchronize(stream);
    float elapsed = timer.stopTimer();

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    return elapsed;
}

// Fixed verification function - using fabsf() instead of abs()
bool verifyResults(float *h_a, float *h_b, float *h_result, int n, const char *method_name)
{
    bool correct = true;
    const float epsilon = 1e-5f;
    int error_count = 0;

    printf("Verifying %s results...\n", method_name);

    for (int i = 0; i < n; i++)
    {
        // Expected: ((a + b) * 2.0)^2
        float expected = (h_a[i] + h_b[i]) * 2.0f;
        expected = expected * expected;

        // BUG FIX: Use fabsf() instead of abs() for floating point comparison
        if (fabsf(h_result[i] - expected) > epsilon)
        {
            if (error_count < 10) // Limit error reporting
            {
                printf("  Mismatch at index %d: got %.6f, expected %.6f, diff: %.6f\n", i, h_result[i], expected,
                       fabsf(h_result[i] - expected));
            }
            correct = false;
            error_count++;
        }
    }

    if (error_count > 0)
    {
        printf("  Total errors found: %d out of %d elements\n", error_count, n);
    }

    return correct;
}

// Compare two result arrays with detailed analysis
bool compareResults(float *result1, float *result2, int n, const char *name1, const char *name2)
{
    bool match = true;
    const float epsilon = 1e-5f;
    int mismatch_count = 0;
    float max_diff = 0.0f;

    printf("Comparing %s vs %s results...\n", name1, name2);

    for (int i = 0; i < n; i++)
    {
        // BUG FIX: Use fabsf() instead of abs() for floating point comparison
        float diff = fabsf(result1[i] - result2[i]);

        if (diff > max_diff)
        {
            max_diff = diff;
        }

        if (diff > epsilon)
        {
            if (mismatch_count < 10) // Limit error reporting
            {
                printf("  Mismatch at index %d: %s=%.6f, %s=%.6f, diff=%.6f\n", i, name1, result1[i], name2, result2[i],
                       diff);
            }
            match = false;
            mismatch_count++;
        }
    }

    printf("  Maximum difference: %.6f\n", max_diff);
    if (mismatch_count > 0)
    {
        printf("  Total mismatches: %d out of %d elements\n", mismatch_count, n);
    }

    return match;
}

// Debug function to print memory state
void debugMemoryState(float *h_a, float *h_b, float *h_result, int start_idx, int count, const char *label)
{
    printf("\n%s - Memory State (elements %d to %d):\n", label, start_idx, start_idx + count - 1);
    printf("Index |   A     |   B     | Result  | Expected\n");
    printf("------|---------|---------|---------|----------\n");

    for (int i = start_idx; i < start_idx + count && i < VECTOR_SIZE; i++)
    {
        float expected = (h_a[i] + h_b[i]) * 2.0f;
        expected = expected * expected;
        printf("%5d | %7.6f | %7.6f | %7.6f | %8.6f\n", i, h_a[i], h_b[i], h_result[i], expected);
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
    float *h_a = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_b = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_result_traditional = (float *)malloc(VECTOR_SIZE * sizeof(float));
    float *h_result_graph = (float *)malloc(VECTOR_SIZE * sizeof(float));

    // Initialize input data
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        h_a[i] = (float)i / 1000.0f;
        h_b[i] = (float)(i + 1) / 1000.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c, *d_temp;
    cudaMalloc(&d_a, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_b, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_c, VECTOR_SIZE * sizeof(float));
    cudaMalloc(&d_temp, VECTOR_SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, VECTOR_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Run traditional kernel launches
    printf("Running traditional kernel launches...\n");
    float time_traditional = runTraditionalLaunches(d_a, d_b, d_c, d_temp, VECTOR_SIZE);
    cudaMemcpy(h_result_traditional, d_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Run CUDA Graphs
    printf("Running CUDA Graphs...\n");
    float time_graph = runCudaGraphs(d_a, d_b, d_c, d_temp, VECTOR_SIZE);
    cudaMemcpy(h_result_graph, d_c, VECTOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Traditional launches: %8.2f ms\n", time_traditional);
    printf("CUDA Graphs:         %8.2f ms\n", time_graph);
    printf("Speedup:             %8.2fx\n", time_traditional / time_graph);
    printf("Overhead reduction:  %8.2f%%\n", 100.0f * (time_traditional - time_graph) / time_traditional);

    // Verify correctness with fixed verification function
    printf("\nDetailed Verification (FIXED):\n");
    bool traditional_correct = verifyResults(h_a, h_b, h_result_traditional, VECTOR_SIZE, "Traditional");
    bool graph_correct = verifyResults(h_a, h_b, h_result_graph, VECTOR_SIZE, "Graph");

    printf("\nFinal Results:\n");
    printf("Traditional results: %s\n", traditional_correct ? "CORRECT" : "INCORRECT");
    printf("Graph results:       %s\n", graph_correct ? "CORRECT" : "INCORRECT");

    // Compare results between methods with fixed comparison
    printf("\nResult Comparison (FIXED):\n");
    bool results_match = compareResults(h_result_traditional, h_result_graph, VECTOR_SIZE, "Traditional", "Graph");
    printf("Results match:       %s\n", results_match ? "YES" : "NO");

    // Debug memory state for first few elements
    debugMemoryState(h_a, h_b, h_result_traditional, 0, 5, "Traditional Results");
    debugMemoryState(h_a, h_b, h_result_graph, 0, 5, "Graph Results");

    // Check for edge cases - test with some specific indices
    printf("\nEdge Case Analysis:\n");
    int test_indices[] = {0, 1, VECTOR_SIZE / 2, VECTOR_SIZE - 2, VECTOR_SIZE - 1};
    int num_test_indices = sizeof(test_indices) / sizeof(test_indices[0]);

    for (int i = 0; i < num_test_indices; i++)
    {
        int idx = test_indices[i];
        if (idx < VECTOR_SIZE)
        {
            float expected = (h_a[idx] + h_b[idx]) * 2.0f;
            expected = expected * expected;
            printf("Index %d: Traditional=%.6f, Graph=%.6f, Expected=%.6f\n", idx, h_result_traditional[idx],
                   h_result_graph[idx], expected);
        }
    }

    printf("\nBug Analysis Summary:\n");
    printf("1. Fixed abs() -> fabsf() for float comparisons\n");
    printf("2. Added detailed error reporting and analysis\n");
    printf("3. Implemented comprehensive result comparison\n");
    printf("4. Added edge case testing\n");

    // Cleanup
    free(h_a);
    free(h_b);
    free(h_result_traditional);
    free(h_result_graph);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_temp);

    return 0;
}
