#define LEET_GPU_NOIMPORT
#ifndef LEET_GPU_NOIMPORT
#include "solve.h"
#include <cuda_runtime.h>
#endif
#include <cfloat>
#include <cmath>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#ifndef LEET_GPU_NOIMPORT
__global__
#endif
    void
    AttentionKernel(const float *Query, const float *Key, const float *Value, float *Out, int M, int N, int D)
{
    // Let's start with Q*K.T * V - this will result in a MxN matrix that we can output to the console
    // and start working on.
    //
    // Output will be Mxd
    //
    // Do I need to materialize the internal matrix?

    // Considering here that I need to allocate a temporary matrix of MxN to hold the value of Q*K.T
    float *Internal = (float *)malloc(sizeof(float) * M * N);

    for (int I = 0; I < M; I++)
    {
        for (int J = 0; J < N; J++)
        {
            float Sum = 0.0f;
            for (int K = 0; K < D; K++)
            {
                Sum += Query[I * D + K] * Key[J * D + K];
            }
            Internal[I * N + J] = Sum / sqrtf((float)D);
        }
    }

    // Now we need to calculate modify internal such that it has the softmax value
    // for softmax we will use the f(x_i) = e^x_i-max_x/ sum(e^x_i-max_x) for all x
    // So we need to iterate over the three times:
    // 1. Calculate the max value
    // 2. Calculate the exponential sum
    // 3. Divide each element by the exponential sum

    for (float *Row = Internal; (Row - Internal) < M * N; Row += N)
    {
        float MaxValue = -FLT_MAX;
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            MaxValue = MAX(MaxValue, *Col);
        }
        float ExpSum = 0.0f;
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            *Col = expf(*Col - MaxValue);
            ExpSum += *Col;
        }
        for (float *Col = Row; (Col - Row) < N; Col++)
        {
            *Col /= ExpSum;
        }
    }

    // Now we just need to do another matmul to get to the output value
    for (int I = 0; I < M; I++)
    {
        for (int J = 0; J < D; J++)
        {
            float Sum = 0.0f;
            for (int K = 0; K < N; K++)
            {
                Sum += Internal[I * N + K] * Value[K * D + J];
            }
            Out[I * D + J] = Sum;
        }
    }

    if (Internal)
    {
        free(Internal);
    }
}

#ifndef LEET_GPU_NOIMPORT
// Q, K, V, output are device pointers
void solve(const float *Q, const float *K, const float *V, float *output, int M, int N, int d)
{
    AttentionKernel<<<1, 1>>>(Q, K, V, output, M, N, d);
}
#endif
