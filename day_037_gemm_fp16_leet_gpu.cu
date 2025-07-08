#include "solve.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void GEMM_FP16(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    int Col = blockDim.x * blockIdx.x + threadIdx.x;
    int Row = blockDim.y * blockIdx.y + threadIdx.y;

    if (Col < N && Row < M)
    {
        float Res = 0.0f;
        for (int k = 0; k < K; k++)
        {
            Res += __half2float(A[Row * K + k]) * __half2float(B[Col + k * N]);
        }
        Res *= alpha;
        Res += beta * __half2float(C[Row * N + Col]);
        C[Row * N + Col] = __float2half(Res);
    }
}

// A, B, and C are device pointers
void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta)
{
    dim3 BlockDim(32, 32);
    dim3 GridDim((M + 32 - 1) / 32, (N + 32 - 1) / 32);
    GEMM_FP16<<<GridDim, BlockDim>>>(A, B, C, M, N, K, alpha, beta);
}
