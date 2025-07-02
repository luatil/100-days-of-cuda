#include "solve.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define SOLUTION 2

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
#endif

// logits, true_labels, loss are device pointers
void solve(const float *logits, const int *true_labels, float *loss, int N, int C)
{
    printf("Using solution %d\n", SOLUTION);

#if SOLUTION == 1
    CategoricalCrossEntropyLoss<<<1, 1>>>(logits, true_labels, loss, N, C);
#elif SOLUTION == 2
    CategoricalCrossEntropyLoss<<<N, 1>>>(logits, true_labels, loss, N, C);
#endif
}
