#include <cuda_runtime.h>

__global__ void SiLU(const float *Input, float *Output, int N)
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < N)
    {
        float X = Input[Tid];
        float Sigma = 1 / (1 + expf(-X));
        Output[Tid] = X * Sigma;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int BlockDim = 256;
    int GridDim = (N + BlockDim - 1) / BlockDim;
    SiLU<<<GridDim, BlockDim>>>(input, output, N);
}
