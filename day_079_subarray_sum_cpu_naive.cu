#include "day_079_subarray_sum.cuh"

extern "C" void solve_subarray_sum_cpu_naive(const int* input, int* output, int N, int S, int E) {
  for (int I = S; I <= E; I++)
  {
    *output += input[I];
  }
}

// Legacy alias for backwards compatibility
extern "C" void solve_subarray_sum(const int* input, int* output, int N, int S, int E) {
  solve_subarray_sum_cpu_naive(input, output, N, S, E);
}
