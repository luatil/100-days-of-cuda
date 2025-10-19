#include "day_079_subarray_sum.cuh"
#include <stdio.h>
#include <cuda_runtime.h>

// Input: input = [1, 2, 1, 3, 4], S = 1, E = 3
// Output: output = 6

int main() {
  const int Input[] = {1, 2, 1, 3, 4};
  const int N = 5;
  const int S = 1;
  const int E = 3;

  printf("Input array: ");
  for (int I = 0; I < N; I++) {
    printf("%d ", Input[I]);
  }
  printf("\n");
  printf("Computing sum from index %d to %d\n\n", S, E);

  // Test CPU implementation
  int OutputCPU = 0;
  solve_subarray_sum_cpu_naive(Input, &OutputCPU, N, S, E);
  printf("CPU Naive result: %d\n", OutputCPU);

  // Test GPU implementation
  int OutputGPU = 0;
  solve_subarray_sum_gpu_v1_naive(Input, &OutputGPU, N, S, E);
  printf("GPU v1 Naive result: %d\n", OutputGPU);

  // Verify results match
  if (OutputCPU == OutputGPU) {
    printf("\n✓ Results match!\n");
  } else {
    printf("\n✗ Results differ: CPU=%d, GPU=%d\n", OutputCPU, OutputGPU);
  }

  return 0;
}
