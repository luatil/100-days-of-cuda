#ifndef DAY_079_SUBARRAY_SUM_CUH
#define DAY_079_SUBARRAY_SUM_CUH

// CPU implementations
extern "C" void solve_subarray_sum_cpu_naive(const int* input, int* output, int N, int S, int E);

// GPU implementations
extern "C" void solve_subarray_sum_gpu_v1_naive(const int* input, int* output, int N, int S, int E);

// Legacy alias for backwards compatibility
extern "C" void solve_subarray_sum(const int* input, int* output, int N, int S, int E);

#endif // DAY_079_SUBARRAY_SUM_CUH
