#include "day_079_subarray_sum.cuh"
#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>
#include <vector>

// Helper function to measure execution time
template<typename Func>
double benchmark(Func func, int iterations = 100) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    func();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration = end - start;
  return duration.count() / iterations;
}

int main() {
  // Test with varying array sizes
  std::vector<int> sizes = {100, 1000, 10000, 100000, 1000000};

  printf("Subarray Sum Performance Comparison\n");
  printf("====================================\n\n");

  for (int size : sizes) {
    // Create test data
    std::vector<int> input(size);
    for (int i = 0; i < size; i++) {
      input[i] = i % 100;
    }

    int S = 0;
    int E = size - 1;

    // Benchmark CPU implementation
    int outputCPU = 0;
    double timeCPU = benchmark([&]() {
      outputCPU = 0;
      solve_subarray_sum_cpu_naive(input.data(), &outputCPU, size, S, E);
    });

    // Benchmark GPU implementation
    int outputGPU = 0;
    double timeGPU = benchmark([&]() {
      outputGPU = 0;
      solve_subarray_sum_gpu_v1_naive(input.data(), &outputGPU, size, S, E);
    });

    // Verify correctness
    bool correct = (outputCPU == outputGPU);

    printf("Size: %7d elements\n", size);
    printf("  CPU Naive:    %8.4f ms  (result: %d)\n", timeCPU, outputCPU);
    printf("  GPU v1 Naive: %8.4f ms  (result: %d)\n", timeGPU, outputGPU);
    printf("  Speedup:      %8.2fx\n", timeCPU / timeGPU);
    printf("  Correctness:  %s\n", correct ? "✓ PASS" : "✗ FAIL");
    printf("\n");
  }

  return 0;
}
