/**
 * Computes the softmax of two vectors A and B.
 *
 *
 * Input:  A = [1.0, 2.0, 3.0, 4.0]
 * B = [5.0, 6.0, 7.0, 8.0]
 * Output: result = 70.0  (1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0)
 *
 *
 */
#include <stdio.h>
#define SOLUTION_00 0
#define SOLUTION_01 0
#define SOLUTION_02 1

#if SOLUTION_00
// Naive version with just <<<1,1>>> launch
__global__ void Kernel_DotProduct(const float *A, const float *B, float *result, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float Sum = 0.0f;
    for (int i = 0; i < N; i++)
    {
        Sum += A[i] * B[i];
    }
    *result = Sum;
}
// A, B, result are device pointers
void solve(const float *A, const float *B, float *result, int N)
{
    Kernel_DotProduct_00<<<1, 1>>>(A, B, result, N);
}
#elif SOLUTION_01
#define BLOCK_DIM 1024
// Version with <<<1, 1024>>>
__global__ void Kernel_DotProduct(const float *A, const float *B, float *result, int N)
{
    __shared__ float SharedMem[BLOCK_DIM];

    int tx = threadIdx.x;

    // ThreadCoarsening
    float Sum = 0.0f;
    for (int i = 0; i < ((N + blockDim.x - 1) / blockDim.x); i++)
    {
        int pos = blockDim.x * i + tx;
        if (pos < N)
        {
            Sum += A[pos] * B[pos];
        }
    }
    SharedMem[tx] = Sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            SharedMem[tx] += SharedMem[tx + s];
        }
    }
    __syncthreads();

    if (tx == 0)
    {
        *result = SharedMem[0];
    }
}

#define BLOCK_DIM 512
#define COARSE_FACTOR 4

__global__ void KernelDotProduct02(const float *A, const float *B, float *Result, int N)
{
    __shared__ float SharedMem[BLOCK_DIM];

    int Tid = COARSE_FACTOR * blockDim.x * blockIdx.x + threadIdx.x;
    int Tx = threadIdx.x;

    // ThreadCoarsening
    float Sum = 0.0f;
    for (int I = 0; I < COARSE_FACTOR; I++)
    {
        int Pos = Tid + blockDim.x * I;
        if (Pos < N)
        {
            Sum += A[Pos] * B[Pos];
        }
    }
    SharedMem[Tx] = Sum;
    __syncthreads();

    for (int S = blockDim.x / 2; S > 0; S >>= 1)
    {

        if (Tx < S)
        {
            SharedMem[Tx] += SharedMem[Tx + S];
        }
        __syncthreads();
    }

    if (Tx == 0)
    {
        atomicAdd(Result, SharedMem[0]);
    }
}

void Solve(const float *A, const float *B, float *Result, int N)
{
    int GridDim = (N + (BLOCK_DIM * COARSE_FACTOR) - 1) / (BLOCK_DIM * COARSE_FACTOR);
    KernelDotProduct02<<<GridDim, BLOCK_DIM>>>(A, B, Result, N);
}
#endif

int main()
{
}
