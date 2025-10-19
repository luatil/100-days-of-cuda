#ifndef DAY_079_SUBARRAY_SUM_CU
#define DAY_079_SUBARRAY_SUM_CU


extern "C" void solve_subarray_sum(const int* input, int* output, int N, int S, int E) {
  for (int I = S; I <= E; I++)
  {
    *output += input[I];
  }
}

#endif // DAY_079_SUBARRAY_SUM_CU
