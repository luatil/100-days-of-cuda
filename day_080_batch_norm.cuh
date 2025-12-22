#ifndef DAY_080_BATCH_NORM_CUH
#define DAY_080_BATCH_NORM_CUH

extern "C" void batch_norm_cuda_impl(const float* input, const float* gamma, const float* beta, float* output, int N, int C, float eps);

#endif // DAY_080_BATCH_NORM_CUH

