#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define SOLUTION 3

#if SOLUTION == 1
// <1,1> wip version
__global__ void CategoricalCrossEntropyLoss(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    *loss = 0.0f;
    for (int j = 0; j < N; j++)
    {
        float Loss_j = 0.0f;
        for (int k = 0; k < C; k++)
        {
            Loss_j += expf(logits[j * C + k]);
        }
        Loss_j = logf(Loss_j);
        Loss_j -= logits[j * C + true_labels[j]];
        *loss += Loss_j;
    }
    *loss /= N;
}
#elif SOLUTION == 2
// <N,1> wip version
// In this solution each thread calculates Loss_j
// And we use atomicAdd on loss with Loss_j / N
__global__ void CategoricalCrossEntropyLoss(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    int j = blockIdx.x;
    float Loss_j = 0.0f;
    for (int k = 0; k < C; k++)
    {
        Loss_j += expf(logits[j * C + k]);
    }
    Loss_j = logf(Loss_j);
    Loss_j -= logits[j * C + true_labels[j]];
    atomicAdd(loss, Loss_j / N);
}
#elif SOLUTION == 3
// <N,C> version
// Each thread handles one element, block reduces to compute softmax denominator
__global__ void CategoricalCrossEntropyLoss(const float *Logits, const int *TrueLabels, float *Loss, int N, int C)
{
    int J = blockIdx.x;  // sample index
    int K = threadIdx.x; // class index

    __shared__ float SharedExp[1024]; // assuming C <= 1024

    // Each thread computes exp(logits[j*C + k])
    float ExpVal = 0.0f;
    if (K < C)
    {
        ExpVal = expf(Logits[J * C + K]);
    }
    SharedExp[K] = ExpVal;

    __syncthreads();

    // Block reduction to compute sum of exponentials
    for (int Stride = 1; Stride < blockDim.x; Stride *= 2)
    {
        if (K % (2 * Stride) == 0 && K + Stride < C)
        {
            SharedExp[K] += SharedExp[K + Stride];
        }
        __syncthreads();
    }

    // Thread 0 computes the loss for this sample
    if (K == 0)
    {
        float LossJ = logf(SharedExp[0]) - Logits[J * C + TrueLabels[J]];
        atomicAdd(Loss, LossJ / N);
    }
}
#endif

// logits, true_labels, loss are device pointers
void Solve(const float *Logits, const int *TrueLabels, float *Loss, int N, int C)
{
    printf("Using solution %d\n", SOLUTION);

#if SOLUTION == 1
    CategoricalCrossEntropyLoss<<<1, 1>>>(logits, true_labels, loss, N, C);
#elif SOLUTION == 2
    CategoricalCrossEntropyLoss<<<N, 1>>>(logits, true_labels, loss, N, C);
#elif SOLUTION == 3
    CategoricalCrossEntropyLoss<<<N, C>>>(Logits, TrueLabels, Loss, N, C);
#endif
}