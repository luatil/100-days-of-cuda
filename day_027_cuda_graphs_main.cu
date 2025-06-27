#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

#define BLOCK_SIZE 256
#define VECTOR_SIZE (1024 * 1024)  // 1M elements
#define NUM_ITERATIONS 1000

// Simple vector addition kernel
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Vector scaling kernel
__global__ void vectorScale(float* a, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scale;
    }
}

// Vector square kernel
__global__ void vectorSquare(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * a[idx];
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

// Traditional kernel launches
float runTraditionalLaunches(float* d_a, float* d_b, float* d_c, float* d_temp, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDATimer timer;
    
    timer.startTimer();
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
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
float runCudaGraphs(float* d_a, float* d_b, float* d_c, float* d_temp, int n) {
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
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
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

// Verify results are correct
bool verifyResults(float* h_a, float* h_b, float* h_result, int n) {
    bool correct = true;
    const float epsilon = 1e-5f;
    
    for (int i = 0; i < n; i++) {
        // Expected: ((a + b) * 2.0)^2
        float expected = (h_a[i] + h_b[i]) * 2.0f;
        expected = expected * expected;
        
        if (abs(h_result[i] - expected) > epsilon) {
            printf("Mismatch at index %d: got %f, expected %f\n", 
                   i, h_result[i], expected);
            correct = false;
            if (i > 10) break; // Don't spam too many errors
        }
    }
    
    return correct;
}

int main() {
    printf("CUDA Graphs Example - Day 027\n");
    printf("Vector size: %d elements\n", VECTOR_SIZE);
    printf("Iterations: %d\n", NUM_ITERATIONS);
    printf("============================================\n");
    
    // Allocate host memory
    float* h_a = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_b = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_result_traditional = (float*)malloc(VECTOR_SIZE * sizeof(float));
    float* h_result_graph = (float*)malloc(VECTOR_SIZE * sizeof(float));
    
    // Initialize input data
    for (int i = 0; i < VECTOR_SIZE; i++) {
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
    
    // Print results
    printf("\nPerformance Results:\n");
    printf("Traditional launches: %8.2f ms\n", time_traditional);
    printf("CUDA Graphs:         %8.2f ms\n", time_graph);
    printf("Speedup:             %8.2fx\n", time_traditional / time_graph);
    printf("Overhead reduction:  %8.2f%%\n", 
           100.0f * (time_traditional - time_graph) / time_traditional);
    
    // Verify correctness
    printf("\nVerification:\n");
    bool traditional_correct = verifyResults(h_a, h_b, h_result_traditional, VECTOR_SIZE);
    bool graph_correct = verifyResults(h_a, h_b, h_result_graph, VECTOR_SIZE);
    
    printf("Traditional results: %s\n", traditional_correct ? "CORRECT" : "INCORRECT");
    printf("Graph results:       %s\n", graph_correct ? "CORRECT" : "INCORRECT");
    
    // Check if both methods produce the same results
    bool results_match = true;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        if (abs(h_result_traditional[i] - h_result_graph[i]) > 1e-5f) {
            results_match = false;
            break;
        }
    }
    printf("Results match:       %s\n", results_match ? "YES" : "NO");
    
    // Show example values
    printf("\nSample Results (first 5 elements):\n");
    printf("Index | Input A | Input B | Traditional | Graph\n");
    printf("------|---------|---------|-------------|-------\n");
    for (int i = 0; i < 5; i++) {
        printf("%5d | %7.3f | %7.3f | %11.3f | %7.3f\n", 
               i, h_a[i], h_b[i], h_result_traditional[i], h_result_graph[i]);
    }
    
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