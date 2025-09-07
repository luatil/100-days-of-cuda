#include <cuda_runtime.h>
#include <stdio.h>

__global__ void SWiGLU(const float* Input, float* Output, int HalfN) 
{
    int Tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (Tid < HalfN)
    {
        float X1 = Input[Tid];
        float X2 = Input[Tid + HalfN]; // N is guaranteed to be even
        float Sigma = 1 / (1 + expf(-X1));
        float SiLU = X1 * Sigma;
        Output[Tid] = SiLU * X2;
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    SWiGLU<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}
